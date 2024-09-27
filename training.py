import wandb
from src.environments.goright import GoRightEnv
from src.agents.q_learning import QLearning
import numpy as np
from multiprocessing import Process, Manager
import random
import os
import yaml


def train_agent(seed, config, return_dict):
    try:
        import numpy as np
        from src.environments.goright import GoRightEnv
        from src.agents.q_learning import QLearning
        import random
        import traceback
        import wandb
        import os
        import copy

        # Set the random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Extract configurations
        training_loops = config["training"]["training_loops"]
        n_steps = config["training"]["n_steps"]
        learning_rate = config["agent"]["learning_rate"]
        discount_factor = config["agent"]["discount_factor"]
        num_prize_indicators = config["environment"]["num_prize_indicators"]
        rollout_length = config["agent"]["rollout_length"]
        status_intensities = config["environment"]["status_intensities"]
        env_length = config["environment"]["env_length"]
        is_observation_noisy = config["environment"]["is_observation_noisy"]
        use_value_expansion = config["training"]["use_value_expansion"]

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
            group=config["wandb"]["group_name"]
        )

        # Initialize the environment and agent
        base_env = GoRightEnv(
            is_observation_noisy=is_observation_noisy,
            seed=seed,
            num_prize_indicators=num_prize_indicators,
            length=env_length,
            status_intensities=status_intensities
        )

        agent = QLearning(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            env_action_space=base_env.action_space,
            rollout_length=rollout_length,
            environment_length=env_length,
            intensities_length=len(status_intensities),
            num_prize_indicators=num_prize_indicators,
        )

        # Initialize an array to store returns over time
        returns_over_time = np.zeros((training_loops, n_steps))

        for training_step in range(training_loops):
            obs, info = base_env.reset()

            # Perform one training episode
            train_total_reward = 0
            agent.td_error = []  # Reset TD errors for this episode
            for _ in range(n_steps):
                action = agent.get_action(obs, greedy=False)
                next_obs, reward, terminated, truncated, info = base_env.step(action)
                agent.update_q_values(
                    obs,
                    action,
                    reward,
                    next_obs,
                    # use_value_expansion=use_value_expansion,
                    # dynamics_model=copy.deepcopy(base_env),
                )
                obs = next_obs
                train_total_reward += reward

            # Evaluation episode
            obs, info = base_env.reset()

            discounted_return = 0
            eval_total_reward = 0
            gamma = 1.0
            for step in range(n_steps):
                action = agent.get_action(obs, greedy=True)
                next_obs, reward, terminated, truncated, info = base_env.step(action)
                obs = next_obs
                discounted_return += gamma * reward
                gamma *= discount_factor
                eval_total_reward += reward

                wandb.log(
                    {
                        "evaluation/step": step + n_steps * training_step,
                        "evaluation/discounted_return": discounted_return,
                        "evaluation/reward": reward,
                    }
                )

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

        # Create a wandb artifact
        artifact = wandb.Artifact(f"q_values_seed_{seed}", type="model")
        artifact.add_file(q_values_filename)

        # Log the artifact
        wandb.log_artifact(artifact)

        # Remove the local file after logging to wandb
        os.remove(q_values_filename)

        # Finish the wandb run
        wandb.finish()

        # Store the results in the shared dictionary
        return_dict[seed] = {
            "returns_over_time": returns_over_time,
            "q_values": agent.q_values
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

if __name__ == "__main__":
    # Read configurations from YAML file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_seeds = config["training"]["n_seeds"]
    start_seed = config["training"]["start_seed"]

    # Seeds for reproducibility
    seeds = np.arange(start_seed, start_seed+n_seeds)

    # Use a Manager to create a shared dictionary for collecting results
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
    training_loops = config["training"]["training_loops"]
    n_steps = config["training"]["n_steps"]
    returns_over_time = np.zeros((training_loops, n_seeds, n_steps))
    for idx, seed in enumerate(seeds):
        data = return_dict[seed]
        if data.get("error"):
            print(f"Error in process with seed {seed}:\n{data['error']}")
        else:
            returns_over_time[:, idx, :] = data["returns_over_time"]

    with open('returns_over_time.npy', 'wb') as f:
        np.save(f, returns_over_time)
