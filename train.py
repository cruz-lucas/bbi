"""Training script."""

import argparse
import logging
import os
import traceback
from multiprocessing import Process
from typing import List, Tuple

import gin
import goright.env
import gymnasium as gym
import numpy as np
import wandb
from wandb.util import generate_id

import goright
from bbi.agent import BoundingBoxPlanningAgent
from bbi.models import ExpectationModel, ModelBase, PerfectModel, SamplingModel

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
    uncertainty_type: str = "unselective",
    project: str = "BBI",
    notes: str = "",
    group_name: str = "default",
) -> None:
    """Trains a Q-Learning agent using the provided seed and configuration.

    Args:
        seed (int): _description_
        training_loops (int, optional): _description_. Defaults to 600.
        n_steps (int, optional): _description_. Defaults to 500.
        step_size (float, optional): _description_. Defaults to 0.1.
        initial_value (float, optional): _description_. Defaults to 0.0.
        discount (float, optional): _description_. Defaults to 0.9.
        max_horizon (int, optional): _description_. Defaults to 5.
        tau (float, optional): _description_. Defaults to 1.0.
        environment_id (str, optional): _description_. Defaults to "GoRight-v0".
        obs_shape (Tuple[int, int, int, int], optional): _description_. Defaults to (11, 3, 2, 2).
        status_intensities (List[int], optional): _description_. Defaults to [0, 5, 10].
        model_type (str, optional): _description_. Defaults to "expectation".
        uncertainty_type (str, optional): _description_. Defaults to "unselective".
        project (str, optional): _description_. Defaults to "BBI".
        notes (str, optional): _description_. Defaults to "".
        group_name (str, optional): _description_. Defaults to "".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
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
                "uncertainty_type": uncertainty_type,
            },
            reinit=True,
            notes=notes,
            group=group_name,

            dir=f"./wandb_{group_name}_seed_{seed}",
            id=f"{group_name}_seed_{seed}_{generate_id()}",
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
            uncertainty_type=uncertainty_type,
        )

        model_cls = {
            "expectation": ExpectationModel,
            "sampling": SamplingModel,
            "perfect": PerfectModel,
            "none": ModelBase
        }

        if model_type not in model_cls.keys():
            raise ValueError(
                f"model_type must be one of: {model_cls.keys()}. Got: {model_type}"
            )

        model = model_cls[model_type](
            num_prize_indicators=len(obs_shape[2:]),
            env_length=obs_shape[0],
            status_intensities=status_intensities,
            seed=int(seed),
        )
        model.reset()
        prev_status = None  # only used when model_type == perfect

        logger.info(f"Starting training for seed {seed}")

        for training_step in range(training_loops):
            episode_seed = int((seed * 10_000) + training_step)
            obs, info = env.reset(seed=episode_seed)

            # Training episode
            train_total_reward = 0.0
            agent.epsilon = 1.0
            for step in range(n_steps):
                logger.debug(
                    "======================================\n"
                )
                logger.debug(
                    f"Starting - Training step {training_step}, inner step {step}, global step {training_step*500+step}: "
                )
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                if model_type == "perfect":
                    env_unwrapped = env.unwrapped
                    if not (isinstance(env_unwrapped, ModelBase) or isinstance(env_unwrapped, goright.env.GoRight)):
                        raise ValueError(
                            f"Environment must inherit ModelBase. Got: {type(env_unwrapped)}"
                        )
                    env_state = env_unwrapped.state.get_state()
                    prev_status = env_state[1]

                td_error = agent.update(
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    reward=float(reward),
                    model=model,
                    prev_status=prev_status,
                )

                logger.debug(
                    f"Summary: obs={obs} action={action}, next_obs={next_obs}, reward={reward}, TD error={td_error}"
                )

                obs = next_obs
                train_total_reward += float(reward)

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
                discounted_return += discount**step * float(reward)
                eval_total_reward += float(reward)

                wandb.log(
                    {
                        "evaluation/discounted_return": discounted_return,
                        "evaluation/discounted_reward": discount**step * float(reward),
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
                    "train/average_td_error": (
                        np.mean(agent.td_errors) if agent.td_errors else 0
                    ),
                    "train/total_evaluation_reward": eval_total_reward,
                    "train/total_evaluation_discounted_return": discounted_return,
                    "train_step": training_step,
                },
                # step = training_step,
                # commit = True
            )

            logger.debug(f"Training step {training_step}: Train reward={train_total_reward}, Eval reward={eval_total_reward}")

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


def train_agent_wrapper(seed: int, config_file: str):
    """_summary_.

    Args:
        seed (int): _description_
        config_file (str): _description_
    """
    gin.parse_config_file(config_file)
    train_agent(seed)


def run_seeds(n_seeds: int = 100, start_seed: int = 0, config_file: str = "bbi/config/goright_bbi.gin") -> None:
    """Main function to initiate training across multiple seeds.

    Args:
        n_seeds (int, optional): _description_. Defaults to 100.
        start_seed (int, optional): _description_. Defaults to 0.
        config_file (str, optional): _description_. Defaults to "bbi/config/goright_bbi.gin".
    """
    seeds = np.arange(start_seed, start_seed + n_seeds)
    processes = []

    for seed in seeds:
        # train_agent_wrapper(seed=int(seed), config_file=config_file)
        p = Process(
            target=train_agent_wrapper,
            args=(seed, config_file),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train learning agent with GoRight environment."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="goright_bbi",
        help="Path to the config gin file",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of seeds to run",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=99,
        help="Initial seed",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Flag to log updates",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,# if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            # logging.StreamHandler(),
            logging.FileHandler(f"training_{args.config_file}.log", mode="w")
        ]
    )

    run_seeds(n_seeds=args.n_seeds, start_seed=args.start_seed, config_file=f"bbi/config/{args.config_file}.gin")
