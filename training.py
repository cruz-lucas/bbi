
import wandb
from src.environments.goright import GoRightEnv
from src.agents.q_learning import QLearning
import numpy as np
from multiprocessing import Process, Manager, Lock
# from tqdm import tqdm
import random
import os

# def train_agent(seed, training_loops, n_steps, learning_rate, discount_factor, return_dict, lock, position):
def train_agent(seed, training_loops, n_steps, learning_rate, discount_factor, return_dict, num_prize_indicators=2):
    try:
        import numpy as np
        from src.environments.goright import GoRightEnv
        from src.agents.q_learning import QLearning
        # from tqdm import tqdm
        import random
        import traceback
        # import pyinstrument
        import wandb
        import os

        # Start the profiler
        # profiler = pyinstrument.Profiler()
        # profiler.start()

        # Set the random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        env_length = 10
        status_intensities = [0, 5, 10]

        # Initialize wandb for this process
        wandb.init(
            project="BBI_reproduction",
            name=f"GoRight_Q_Learning_Run_seed_{seed}",
            config={
                "seed": seed,
                "training_loops": training_loops,
                "n_steps": n_steps,
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "is_observation_noisy": True,
                "env_length": env_length,
                "num_prize_indicators": num_prize_indicators,
                "status_intensities": status_intensities,
            },
            reinit=True,
            notes='running 10 seeds for 3x10^5 training steps without progress bar and profilling'
        )

        # Initialize the environment and agent
        base_env = GoRightEnv(
            is_observation_noisy=True,
            seed=seed,
            num_prize_indicators=num_prize_indicators,
            length=env_length,
            status_intensities=status_intensities,
        )

        agent = QLearning(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            env_action_space=base_env.action_space
        )

        # Initialize an array to store returns over time
        returns_over_time = np.zeros((training_loops, n_steps))

        # Training loop with progress bar
        # with lock:
        #     pbar = tqdm(total=training_loops, desc=f"Seed {seed}", position=position)

        for training_step in range(training_loops):
            obs, info = base_env.reset(seed=seed)

            # Perform one training episode
            total_reward = 0
            agent.td_error = []  # Reset TD errors for this episode
            for _ in range(n_steps):
                action = agent.get_action(obs, greedy=False)
                next_obs, reward, terminated, truncated, info = base_env.step(action)
                agent.update_q_values(obs, action, reward, next_obs)
                obs = next_obs
                total_reward += reward

            # Log metrics after each training episode
            wandb.log({
                "training_step": training_step,
                "total_reward": total_reward,
                "average_td_error": np.mean(agent.td_error) if agent.td_error else 0
            })

            # Evaluation episode
            obs, info = base_env.reset(seed=seed)

            discounted_return = 0
            for step in range(n_steps):
                action = agent.get_action(obs, greedy=True)
                next_obs, reward, terminated, truncated, info = base_env.step(action)
                obs = next_obs
                discounted_return += (discount_factor ** step) * reward
                returns_over_time[training_step, step] = discounted_return

            # Log evaluation metrics
            wandb.log({
                "evaluation_step": training_step,
                "discounted_return": discounted_return,
            })

        #     # Update the progress bar
        #     with lock:
        #         pbar.update(1)

        # # Close the progress bar
        # with lock:
        #     pbar.close()

        # Save the Q-values table as an artifact
        q_values_filename = f"q_values_seed_{seed}.npz"
        np.savez_compressed(q_values_filename, q_values=agent.q_values)

        # Create a wandb artifact
        artifact = wandb.Artifact(f'q_values_seed_{seed}', type='model')
        artifact.add_file(q_values_filename)

        # Log the artifact
        wandb.log_artifact(artifact)

        # Remove the local file after logging to wandb
        os.remove(q_values_filename)

        # Stop the profiler
        # profiler.stop()

        # Finish the wandb run
        wandb.finish()

        # Store the results and profiling data in the shared dictionary
        return_dict[seed] = {
            'returns_over_time': returns_over_time,
            # 'profile_output': profiler.output_text(unicode=True, color=True)
        }

    except Exception as e:
        # Capture the exception and store it in the return_dict
        error_message = f"Exception in process with seed {seed}:\n{traceback.format_exc()}"
        return_dict[seed] = {
            'returns_over_time': None,
            # 'profile_output': None,
            'error': error_message
        }

if __name__ == "__main__":
    # import pyinstrument
    # from multiprocessing import Lock
    import random

    # Hyperparameters
    training_loops = 300_000
    n_steps = 500
    learning_rate = 0.5
    discount_factor = 0.9
    n_seeds = 10  # Adjust this based on the number of CPU cores you have

    # Seeds for reproducibility
    seeds = np.arange(n_seeds)

    # Use a Manager to create a shared dictionary for collecting results
    manager = Manager()
    return_dict = manager.dict()
    processes = []

    # # Create a lock for tqdm to prevent overlapping output
    # lock = Lock()

    # Start a separate process for each seed
    for idx, seed in enumerate(seeds):
        p = Process(target=train_agent, args=(
            # seed, training_loops, n_steps, learning_rate, discount_factor, return_dict, lock, idx))
            seed, training_loops, n_steps, learning_rate, discount_factor, return_dict))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Aggregate the results from all processes
    returns_over_time = np.zeros((training_loops, n_seeds, n_steps))
    for idx, seed in enumerate(seeds):
        data = return_dict[seed]
        if data.get('error'):
            print(f"Error in process with seed {seed}:\n{data['error']}")
        else:
            returns_over_time[:, idx, :] = data['returns_over_time']
            # profile_output = data['profile_output']
            # print(f"Profiling results for seed {seed}:\n")
            # print(profile_output)
