"""Training script."""

import argparse
import logging
import os
import traceback
from typing import List, Tuple

import gin
import gymnasium as gym
import numpy as np
import wandb

from bbi.agent import BoundingBoxPlanningAgent
from bbi.models import ExpectationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@gin.configurable
def train_agent(
    seed: int,
    training_loops: int = 600,
    n_steps: int = 500,
    step_size: float = 0.1,
    initial_value: float = 0.0,
    discount: float = 0.9,
    max_horizon: int = 5,
    tau: float = 1.0,
    environment_id: str = "GoRight-v0",
    obs_shape: Tuple[int, int, int, int] = (11, 3, 2, 2),
    status_intensities: List[int] = [0, 5, 10],
    model_type: str = "expectation",
    project: str = "BBI",
    notes: str = "",
    group_name: str = "",
) -> None:
    """Trains a Q-Learning agent using the provided seed and configuration.

    Args:
        seed (int): The random seed for reproducibility.
        config (Dict[str, Any]): Configuration parameters for training.
        return_dict (Dict[int, Any]): Shared dictionary to collect results across processes.
    """
    try:
        wandb.init(
            project=project,
            name=f"{group_name}_seed_{seed}",
            config={
                "seed": seed,
                "training_loops": training_loops,
                "n_steps": n_steps,
                "step_size": step_size,
                "initial_value": initial_value,
                "discount": discount,
                "max_horizon": max_horizon,
                "tau": tau,
                "environment_id": environment_id,
                "obs_shape": obs_shape,
                "model_type": model_type,
            },
            reinit=True,
            notes=notes,
            group=group_name,
            dir=f"./wandb_{group_name}_seed_{seed}",
            id=f"{group_name}_seed_{seed}_{wandb.util.generate_id()}",
        )

        env = gym.make(id=environment_id)

        agent = BoundingBoxPlanningAgent(
            state_shape=obs_shape,
            num_actions=2,
            step_size=step_size,
            discount_rate=discount,
            epsilon=1.0,
            horizon=max_horizon,
            initial_value=initial_value,
            seed=seed,
        )

        model = ExpectationModel(
            num_prize_indicators=len(obs_shape[2:]),
            env_length=obs_shape[0],
            status_intensities=status_intensities,
            seed=int(seed),
        )

        if model:
            model.reset()

        logger.info(f"Starting training for seed {seed}")

        for training_step in range(training_loops):
            episode_seed = int((seed * 10_000) + training_step)
            obs, info = env.reset(seed=episode_seed)

            # Training episode
            train_total_reward = 0.0
            agent.epsilon = 1.0
            for step in range(n_steps):
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                agent.update(
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    reward=reward,
                    model=model,
                )

                obs = next_obs
                train_total_reward += reward

                if terminated or truncated:
                    break

            # Evaluation episode
            obs, info = env.reset(seed=episode_seed + 1_000)
            discounted_return = 0.0
            eval_total_reward = 0.0
            agent.epsilon = 0.0
            for step in range(n_steps):
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs
                discounted_return += discount**step * reward
                eval_total_reward += reward

                wandb.log(
                    {
                        "evaluation/discounted_return": discounted_return,
                        "evaluation/discounted_reward": discount**step * reward,
                        "evaluation/reward": reward,
                        "eval_step": step + n_steps * training_step,
                    },
                    # step = step + n_steps * training_step,
                    # commit = False
                )

                if terminated or truncated:
                    break

            wandb.log(
                {
                    "train/total_train_reward": train_total_reward,
                    "train/average_td_error": np.mean(agent.td_errors)
                    if agent.td_errors
                    else 0,
                    "train/total_evaluation_reward": eval_total_reward,
                    "train/total_evaluation_discounted_return": discounted_return,
                    "train_step": training_step,
                },
                # step = training_step,
                # commit = True
            )

        # Save Q-values
        q_values_filename = f"q_values_seed_{seed}.npz"
        np.savez_compressed(q_values_filename, q_values=agent.Q)
        artifact = wandb.Artifact(f"q_values_seed_{seed}", type="model")
        artifact.add_file(q_values_filename)
        wandb.log_artifact(artifact)
        os.remove(q_values_filename)
        wandb.finish()

        logger.info(f"Training completed for seed {seed}")

    except Exception as e:
        error_message = (
            f"Exception in process with seed {seed}: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_message)


@gin.configurable
def run_seeds(n_seeds: int = 100, start_seed: int = 0) -> None:
    """Main function to initiate training across multiple seeds."""

    seeds = np.arange(start_seed, start_seed + n_seeds)
    # processes = []

    for seed in seeds:
        train_agent(seed=seed)
    #     p = Process(
    #         target=train_agent,
    #         args=(seed,),
    #     )
    #     processes.append(p)
    #     p.start()

    # for p in processes:
    #     p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BBI agent with GoRight environment."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="goright_bbi",
        help="Path to the config gin file",
    )

    args = parser.parse_args()

    gin.parse_config_file(f"bbi/config/{args.config_file}.gin")
    run_seeds()
