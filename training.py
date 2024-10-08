import wandb
from src.environments.goright import GoRightEnv
from src.agents.agents import QLearningAgent
import numpy as np
from multiprocessing import Process, Manager
import os
import yaml
import argparse
import traceback
import copy


def train_agent(seed, config, return_dict):
    try:
        training_loops = config["training"]["training_loops"]
        n_steps = config["training"]["n_steps"]
        learning_rate = config["agent"]["learning_rate"]
        gamma = config["agent"]["gamma"]
        num_prize_indicators = config["environment"]["num_prize_indicators"]
        max_horizon = config["agent"]["max_horizon"]
        status_intensities = config["environment"]["status_intensities"]
        env_length = config["environment"]["env_length"]
        has_state_offset = config["environment"]["has_state_offset"]

        # Initialize wandb for this process
        wandb.init(
            project=config["wandb"]["project"],
            name=f"{config['wandb']['run_name']}_seed_{seed}",
            config={
                "seed": seed,
                **config["training"],
                **config["agent"],
                **config["environment"],
            },
            reinit=True,
            notes=config["wandb"]["notes"],
            group=config["wandb"]["group_name"],
        )

        # Initialize the environment and agent
        base_env = GoRightEnv(
            has_state_offset=has_state_offset,
            num_prize_indicators=num_prize_indicators,
            length=env_length,
            status_intensities=status_intensities,
        )
        agent = QLearningAgent(
            gamma=gamma,
            action_space=base_env.action_space,
            max_horizon=max_horizon,
            environment_length=env_length,
            intensities_length=len(status_intensities),
            num_prize_indicators=num_prize_indicators,
        )

        # Initialize an array to store returns over time
        returns_over_time = np.zeros((training_loops, n_steps))

        for training_step in range(training_loops):
            episode_seed = (seed * 10_000) + training_step
            obs, info = base_env.reset(seed=episode_seed)

            # Perform one training episode
            train_total_reward = 0
            agent.td_error = []  # Reset TD errors for this episode
            for step in range(n_steps):
                action = agent.get_action(obs, greedy=False)
                next_obs, reward, terminated, truncated, info = base_env.step(action)
                td_error = agent.update_q_values(
                    obs,
                    action,
                    reward,
                    next_obs,
                    alpha=learning_rate,
                    tau=0,
                    dynamics_model=copy.deepcopy(base_env)
                )
                obs = next_obs
                train_total_reward += reward

                if terminated or truncated:
                    break

            # Evaluation episode
            obs, info = base_env.reset(seed=episode_seed + 1_000)

            discounted_return = 0
            eval_total_reward = 0
            discount = 1
            for step in range(n_steps):
                action = agent.get_action(obs, greedy=True)
                next_obs, reward, terminated, truncated, info = base_env.step(action)
                obs = next_obs
                discounted_return += discount * reward
                discount *= gamma
                eval_total_reward += reward

                wandb.log(
                    {
                        "evaluation/step": step + n_steps * training_step,
                        "evaluation/discounted_return": discounted_return,
                        "evaluation/reward": reward,
                    }
                )

                if terminated or truncated:
                    break

            # Log evaluation metrics
            wandb.log(
                {
                    "train/step": training_step,
                    "train/total_train_reward": train_total_reward,
                    "train/average_td_error": (
                        np.mean(agent.td_error) if agent.td_error else 0
                    ),
                    "train/total_evaluation_reward": eval_total_reward,
                    "train/total_evaluation_discounted_return": discounted_return,
                }
            )

        # Save the Q-values table as an artifact
        q_values_filename = f"q_values_seed_{seed}.npz"
        np.savez_compressed(q_values_filename, q_values=agent.q_values)
        artifact = wandb.Artifact(f"q_values_seed_{seed}", type="model")
        artifact.add_file(q_values_filename)
        wandb.log_artifact(artifact)
        os.remove(q_values_filename)
        wandb.finish()

        # Store the results in the shared dictionary
        return_dict[seed] = {
            "returns_over_time": returns_over_time,
            "q_values": agent.q_values,
        }

    except Exception as e:
        # Capture the exception and store it in the return_dict
        error_message = (
            f"Exception in process with seed {seed}:\n{traceback.format_exc()}"
        )
        return_dict[seed] = {
            "returns_over_time": None,
            "error": error_message,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Train Q-Learning Agent with Hyperparameters"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config YAML file"
    )
    parser.add_argument("--training_loops", type=int, help="Number of training loops")
    parser.add_argument("--n_steps", type=int, help="Number of steps per episode")
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for Q-learning"
    )
    parser.add_argument("--discount_factor", type=float, help="Discount factor (gamma)")
    parser.add_argument("--n_seeds", type=int, help="Number of seeds")
    parser.add_argument("--start_seed", type=int, help="Start seed")

    args = parser.parse_args()

    # Load the config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments if provided
    if args.training_loops is not None:
        config["training"]["training_loops"] = args.training_loops
    if args.n_steps is not None:
        config["training"]["n_steps"] = args.n_steps
    if args.learning_rate is not None:
        config["agent"]["learning_rate"] = args.learning_rate
    if args.discount_factor is not None:
        config["agent"]["discount_factor"] = args.discount_factor
    if args.n_seeds is not None:
        config["training"]["n_seeds"] = args.n_seeds
    if args.start_seed is not None:
        config["training"]["start_seed"] = args.start_seed

    n_seeds = config["training"]["n_seeds"]
    start_seed = config["training"]["start_seed"]

    seeds = np.arange(start_seed, start_seed + n_seeds)

    manager = Manager()
    return_dict = manager.dict()
    processes = []

    # Start a separate process for each seed
    for idx, seed in enumerate(seeds):
        p = Process(
            target=train_agent,
            args=(
                seed,
                config,
                return_dict,
            ),
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Aggregate the results from all processes
    # returns_over_time = np.zeros(
    #     (config["training"]["training_loops"], n_seeds, config["training"]["n_steps"])
    # )
    # for idx, seed in enumerate(seeds):
    #     data = return_dict[seed]
    #     if data.get("error"):
    #         print(f"Error in process with seed {seed}:\n{data['error']}")
    #     else:
    #         returns_over_time[:, idx, :] = data["returns_over_time"]


if __name__ == "__main__":
    main()
