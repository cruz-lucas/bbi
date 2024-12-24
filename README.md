

# Bounding-Box Inference for Error-Aware Model-Based Reinforcement Learning

This repository provides an **unofficial** Python implementation for the paper:

> **Talvitie, Erin J., et al.** “Bounding-Box Inference for Error-Aware Model-Based Reinforcement Learning.” Reinforcement Learning Conference, 2024. [\[Paper Link\]](https://openreview.net/forum?id=dPP1KqRb7l)

Here, we demonstrate how to implement bounding-box inference in model-based RL, using a variety of agents and environment configurations for the "GoRight" task, among others.

---

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Installation](#installation)
3. [Usage](#usage)
   1. [Running the Simulator (Rendering)](#running-the-simulator-rendering)
   2. [Training Agents](#training-agents)
4. [Configuration Files](#configuration-files)
5. [Agents and Models](#agents-and-models)
6. [References](#references)
7. [License](#license)

---

## Repository Structure

```bash
bbi/
├── agents/
│   ├── base_agent.py
│   ├── planning_agent_base.py
│   ├── qlearning.py
│   ├── selective_planning_agent.py
│   └── unselective_planning_agent.py
├── config/
│   └── ... (config files)
├── environments/
│   ├── resources/
│   │   └── ... (images and other assets)
│   ├── base_env.py
│   └── goright.py
├── models/
│   ├── bbi.py
│   ├── expectation.py
│   ├── linear_bbi.py
│   ├── neural_bbi.py
│   ├── regression_tree_bbi.py
│   └── sampling.py
├── utils.py
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
├── README.md
├── train.py
└── uv.lock
```

- **agents/**: Contains various agent implementations (Q-Learning, Q-learning with planning, selective/unselective).
- **config/**: YAML files specifying hyperparameters and environment configurations for different experiments.
- **environments/**:
  - **resources/**: Images and other assets (e.g. sprites for rendering).
  - **base_env.py** and **goright.py**: Environment definitions.
- **models/**: Different approaches (hand-coded bounding-box inference, sampling, neural, linear, regression tree, etc.).
- **utils.py**: Utility functions (such as rendering the simulator, loading configs, etc.).
- **train.py**: Main script for training agents with various configurations.
- **pyproject.toml**, **Makefile**, **uv.lock**: Project metadata, optional build instructions, and lock files.
- **.pre-commit-config.yaml**: Optional git hooks and formatting/linting rules.

---

## Installation

### Using [uv](https://github.com/previm/uv)

If you use [uv](https://github.com/previm/uv) to manage ephemeral virtual environments, you can simply:

1. **Clone** this repository:
   ```bash
   git clone https://github.com/yourusername/bbi.git
   cd bbi
   ```
2. **Run** with `uv`:
   ```bash
   # For example, to run the training script:
   uv run python train.py
   ```
   This automatically creates a temporary environment, installs dependencies (from `pyproject.toml` and `uv.lock`), and executes `train.py`.

### Virtual Environment with UV

If you prefer using **uv** to create a standard virtual environment (rather than running scripts directly via `uv run`), you can set up a persistent environment as follows:

1. **Initialize** a named environment:
   ```bash
   uv init .venv
   ```
   This creates a directory that holds the virtual environment.

2. **Install** dependencies into `.venv`:
   ```bash
   source .venv/bin/activate
   uv sync
   ```
   By default, `uv sync` will look at your `pyproject.toml` (and optionally `uv.lock` if present) to install dependencies.

3. **Run** scripts or Python commands:
   ```bash
   python train.py --config config/goright_q_learning.yaml
   ```

When finished, simply deactivate by closing your shell or using:
```bash
deactivate
```
(to stop using that environment).

### Traditional Virtual Environment

Alternatively, if you prefer a standard Python virtual environment:

1. **Create** and activate:
   ```bash
   python -m venv .venv
   source venv/bin/activate
   ```
2. **Install**:
   ```bash
   pip install -U pip
   pip install .
   ```
   or install dependencies as needed (from a `requirements.txt` if provided).

---

## Usage

### 1. Running the Simulator (Rendering)

For interactive simulation, use the utility functions in `utils.py` (e.g., `render_simulator`, `render_env`). For example:
```bash
uv run python -m bbi.utils
```
In this interactive window:
- **Left / Right arrows** step through the environment.
- **m** switches between environment models (e.g., `GoRight`, `SamplingModel`, etc.).
- **r** resets the current environment.
- **ESC** quits.

### 2. Training Agents

Use `train.py` to train an agent:
```bash
uv run python train.py --config config/goright_q_learning.yaml
```
- Loads the specified config YAML (e.g. `goright_q_learning.yaml`).
- Spawns multiple processes (one per seed).
- Trains the specified agent (Q-Learning, bounding-box inference, sampling, etc.).
- Optionally logs metrics to [Weights & Biases](https://wandb.ai/) if configured in the config file.

**Key parameters** (e.g., number of steps, environment ID, learning rate) come from the YAML file.

---

## Configuration Files

The `config/` folder contains multiple `.yaml` files specifying hyperparameters and environment details. For example:
- **goright_bbi.yaml**: BBI model with the GoRight environment.
- **goright_q_learning.yaml**: Q-Learning approach.
- **goright_sampling_h5.yaml**: Sampling-based with horizon=5.

A config might look like:
```yaml
environment_id: "bbi/goRight-v0"
model_id: "bbi"
agent:
  learning_rate: 0.1
  gamma: 0.99
  ...
training:
  n_seeds: 4
  start_seed: 42
  n_steps: 100
  ...
```
Adjust parameters to suit your experiments.

---

## Agents and Models

- **agents/**:
  - **qlearning.py**: Basic Q-Learning agent.
  - **selective_planning_agent.py** / **unselective_planning_agent.py**: BBI-based or sampling-based planning approaches.
- **models/**:
  - **bbi.py**, **linear_bbi.py**, **neural_bbi.py**, **regression_tree_bbi.py**: Variations of bounding-box inference models (hand-coded, linear, neural, tree).
  - **expectation.py**, **sampling.py**: Expected-outcome or sampling-based approaches for environment model predictions.

---

## References

- **Paper**: Talvitie, Erin J., et al. “Bounding-Box Inference for Error-Aware Model-Based Reinforcement Learning.” *Reinforcement Learning Conference, 2024.* [OpenReview Link](https://openreview.net/forum?id=dPP1KqRb7l)

---

## License

This project is licensed under the [MIT License](LICENSE).