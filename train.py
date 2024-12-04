"""Training script."""
import copy
import logging
import os
import traceback
from multiprocessing import Manager, Process
from typing import Any, Dict

import gymnasium as gym
import numpy as np

import wandb
from bbi.agents import QLearningAgent
from bbi.environments import ENV_CONFIGURATION
from bbi.utils import load_config, parse_args
from bbi.models import ExpectationModel
from bbi.models import SamplingModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_agent(seed: int, config: Dict[str, Any], return_dict: Dict[int, Any]) -> None:
    """Trains a Q-Learning agent using the provided seed and configuration.

    Args:
        seed (int): The random seed for reproducibility.
        config (Dict[str, Any]): Configuration parameters for training.
        return_dict (Dict[int, Any]): Shared dictionary to collect results across processes.
    """
    try:
        # Extract configurations
        training_loops = config["training"]["training_loops"]
        n_steps = config["training"]["n_steps"]
        learning_rate = config["agent"]["learning_rate"]
        gamma = config["agent"]["gamma"]
        max_horizon = config["agent"]["max_horizon"]
        verbose = config["verbose"]
        env_id = config["environment_id"]
        model_id = config["model_id"]
        env_config = ENV_CONFIGURATION.get(env_id, {})

        # Initialize Weights & Biases for logging
        wandb.init(
            project=config["wandb"]["project"],
            name=f"{config['wandb']['group_name']}_seed_{seed}",
            config={
                "seed": seed,
                **config["training"],
                **config["agent"],
                **env_config,
            },
            reinit=True,
            notes=config["wandb"]["notes"],
            group=config["wandb"]["group_name"],
            dir=f"./wandb_{config['wandb']['group_name']}_seed_{seed}",
            id=f"{config['wandb']['group_name']}_seed_{seed}_{wandb.util.generate_id()}",
        )

        # Define custom metrics with their own steps
        # wandb.define_metric("train/*", step_metric="train_step", step_sync=False)
        # wandb.define_metric("evaluation/*", step_metric="eval_step", step_sync=False)

        # Initialize the environment and agent
        env = gym.make(id=env_id)
        agent = QLearningAgent(
            gamma=gamma,
            action_space=env.action_space,
            environment_length=env_config.get("env_length", 11),
            intensities_length=len(env_config.get("status_intensities", [0, 5, 10])),
            num_prize_indicators=env_config.get("num_prize_indicators", 2),
        )

        logger.info(f"Starting training for seed {seed}")

        for training_step in range(training_loops):
            episode_seed = int((seed * 10_000) + training_step)
            obs, info = env.reset(seed=episode_seed)

            # Training episode
            train_total_reward = 0.0
            td_errors = []
            for step in range(n_steps):
                action = agent.get_action(obs, greedy=False)
                next_obs, reward, terminated, truncated, info = env.step(action)

                if model_id == 'perfect':
                    model = copy.deepcopy(env)
                elif model_id == 'expect':                    
                    model = ExpectationModel(
                        num_prize_indicators=env_config.get("num_prize_indicators", 2),
                        env_length=env_config.get("env_length", 11),
                        status_intensities=env_config.get("status_intensities", [0, 5, 10]),
                        has_state_offset=False
                    )
                    model.reset(seed=episode_seed + 1_000)
                    model.set_state(
                        state=env.unwrapped.state,
                        previous_status=env.unwrapped.previous_status
                    )
                elif model_id == 'sampling':
                    model = SamplingModel(
                        num_prize_indicators=env_config.get("num_prize_indicators", 2),
                        env_length=env_config.get("env_length", 11),
                        status_intensities=env_config.get("status_intensities", [0, 5, 10]),
                        has_state_offset=False
                    )
                    model.reset(seed=episode_seed + 1_000)
                    model.set_state(
                        state=env.unwrapped.state,
                        previous_status=env.unwrapped.previous_status
                    )
                else:
                    model = None

                td_error = agent.update_q_values(
                    obs,
                    action,
                    reward,
                    next_obs,
                    alpha=learning_rate,
                    tau=0,
                    max_horizon=max_horizon,
                    dynamics_model=model,
                )
                obs = next_obs
                train_total_reward += reward
                td_errors.append(td_error)

                if terminated or truncated:
                    break

            # Evaluation episode
            obs, info = env.reset(seed=episode_seed + 1_000)
            discounted_return = 0.0
            eval_total_reward = 0.0
            for step in range(n_steps):
                action = agent.get_action(obs, greedy=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs
                discounted_return += gamma**step * reward
                eval_total_reward += reward

                wandb.log(
                    {
                        "evaluation/discounted_return": discounted_return,
                        "evaluation/discounted_reward": gamma**step * reward,
                        "evaluation/reward": reward,
                        "eval_step": step + n_steps * training_step,
                    },
                    # step = step + n_steps * training_step,
                    # commit = False
                )

                if terminated or truncated:
                    break

            # Log training metrics
            wandb.log(
                {
                    "train/total_train_reward": train_total_reward,
                    "train/average_td_error": np.mean(td_errors) if td_errors else 0,
                    "train/total_evaluation_reward": eval_total_reward,
                    "train/total_evaluation_discounted_return": discounted_return,
                    "train_step": training_step,
                },
                # step = training_step,
                # commit = True
            )

            if training_step % verbose == 0:
                logger.info(f"Completed training step {training_step}/{training_loops}, seed {seed}")

        # Save Q-values
        q_values_filename = f"q_values_seed_{seed}.npz"
        np.savez_compressed(q_values_filename, q_values=agent.q_values)
        artifact = wandb.Artifact(f"q_values_seed_{seed}", type="model")
        artifact.add_file(q_values_filename)
        wandb.log_artifact(artifact)
        os.remove(q_values_filename)
        wandb.finish()

        logger.info(f"Training completed for seed {seed}")

        # Store results
        return_dict[seed] = {
            "q_values": agent.q_values,
        }

    except Exception as e:
        # Handle exceptions
        error_message = f"Exception in process with seed {seed}: {str(e)}\n{traceback.format_exc()}"
        return_dict[seed] = {
            "error": error_message,
        }
        logger.error(error_message)


def main() -> None:
    """Main function to initiate training across multiple seeds."""
    args = parse_args()
    config = load_config(args)

    n_seeds = config["training"]["n_seeds"]
    start_seed = config["training"]["start_seed"]
    seeds = np.arange(start_seed, start_seed + n_seeds)

    manager = Manager()
    return_dict = manager.dict()
    processes = []

    # Start processes for each seed
    for seed in seeds:
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

    # Join processes
    for p in processes:
        p.join()

    # Optionally, process the results in return_dict
    for seed in seeds:
        result = return_dict.get(seed)
        if result is not None:
            if "error" in result:
                logger.error(f"Seed {seed} encountered an error: {result['error']}")
            else:
                logger.info(f"Seed {seed} completed successfully.")


if __name__ == "__main__":
    main()
