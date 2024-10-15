import streamlit as st
import subprocess
import yaml
import os
import uuid
from pathlib import Path
import time

def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f)

def load_default_config(default_path='config.yaml'):
    with open(default_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    st.title("Training Controller for Q-Learning Agent")

    st.header("Set Hyperparameters")

    # Load default config
    default_config = load_default_config()

    # Training Parameters
    st.subheader("Training Parameters")
    training_loops = st.number_input("Number of Training Loops", 
                                     min_value=1, 
                                     value=default_config["training"]["training_loops"], 
                                     step=1)
    n_steps = st.number_input("Number of Steps per Episode", 
                              min_value=1, 
                              value=default_config["training"]["n_steps"], 
                              step=1)
    n_seeds = st.number_input("Number of Seeds", 
                              min_value=1, 
                              value=default_config["training"]["n_seeds"], 
                              step=1)
    start_seed = st.number_input("Start Seed", 
                                 min_value=0, 
                                 value=default_config["training"]["start_seed"], 
                                 step=1)

    # Agent Parameters
    st.subheader("Agent Parameters")
    learning_rate = st.number_input("Learning Rate", 
                                    min_value=0.0, 
                                    value=default_config["agent"]["learning_rate"], 
                                    step=0.01, format="%.4f")
    gamma = st.number_input("Discount Factor (gamma)", 
                                     min_value=0.0, 
                                     max_value=1.0, 
                                     value=default_config["agent"]["gamma"], 
                                     step=0.01, format="%.2f")
    max_horizon = st.number_input("Max Horizon", 
                                  min_value=1, 
                                  value=default_config["agent"]["max_horizon"], 
                                  step=1)

    # Environment Parameters
    st.subheader("Environment Parameters")
    has_state_offset = st.selectbox("Has State Offset", 
                                     options=[True, False], 
                                     index=int(default_config["environment"]["has_state_offset"]))
    env_length = st.number_input("Environment Length", 
                                 min_value=1, 
                                 value=default_config["environment"]["env_length"], 
                                 step=1)
    num_prize_indicators = st.number_input("Number of Prize Indicators", 
                                          min_value=1, 
                                          value=default_config["environment"]["num_prize_indicators"], 
                                          step=1)
    status_intensities = st.text_input("Status Intensities (comma-separated)", 
                                       value=','.join(map(str, default_config["environment"]["status_intensities"])))

    # Wandb Parameters
    st.subheader("Weights & Biases (Wandb) Parameters")
    wandb_project = st.text_input("Wandb Project Name", 
                                  value=default_config["wandb"]["project"])
    wandb_notes = st.text_area("Wandb Notes", 
                               value=default_config["wandb"]["notes"])
    wandb_group_name = st.text_input("Wandb Group Name", 
                                     value=default_config["wandb"]["group_name"])

    # Parse status intensities
    try:
        status_intensities_list = [int(x.strip()) for x in status_intensities.split(',')]
    except ValueError:
        st.error("Status Intensities must be a comma-separated list of integers.")
        status_intensities_list = default_config["environment"]["status_intensities"]

    # Collect all configurations
    if st.button("Start Training"):
        # Create a unique config file to prevent conflicts
        config_id = uuid.uuid4().hex
        config_filename = f"logs/config_{config_id}.yaml"
        config_path = Path(config_filename)

        # Update the config with user inputs
        new_config = {
            "training": {
                "training_loops": int(training_loops),
                "n_steps": int(n_steps),
                "n_seeds": int(n_seeds),
                "start_seed": int(start_seed)
            },
            "agent": {
                "learning_rate": float(learning_rate),
                "gamma": float(gamma),
                "max_horizon": int(max_horizon)
            },
            "environment": {
                "has_state_offset": bool(has_state_offset),
                "env_length": int(env_length),
                "num_prize_indicators": int(num_prize_indicators),
                "status_intensities": status_intensities_list
            },
            "wandb": {
                "project": wandb_project,
                "run_name": wandb_group_name,
                "notes": wandb_notes,
                "group_name": wandb_group_name
            }
        }

        # Save the new config to a YAML file
        save_config(new_config, config_filename)
        st.success(f"Configuration saved to {config_filename}")

        # Start the training process
        st.info("Starting training...")

        # Run the training script as a subprocess
        # Redirect stdout and stderr to log files
        log_filename = f"logs/training_log_{config_id}.txt"
        with open(log_filename, 'w') as log_file:
            process = subprocess.Popen(
                ["python", "training.py", "--config", config_filename],
                stdout=log_file,
                stderr=log_file
            )

        st.success(f"Training started with PID {process.pid}. Logs are being written to {log_filename}")

        st.markdown(f"[View Logs]({os.path.abspath(log_filename)})")

        st.info("Training is running. Below are the real-time logs:")
        log_path = os.path.abspath(log_filename)
        log_container = st.empty()

        try:
            with open(log_path, 'r') as log_file_stream:
                # Continuously read new lines from the log file
                while True:
                    where = log_file_stream.tell()
                    line = log_file_stream.readline()
                    if not line:
                        time.sleep(1)
                        log_file_stream.seek(where)
                    else:
                        log_container.text(line)
        except Exception as e:
            st.error(f"Error reading log file: {e}")

if __name__ == "__main__":
    main()
