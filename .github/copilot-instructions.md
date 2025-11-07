# Copilot Instructions for Pybullet-Gym-Drone

## Project Overview
- This project simulates drone navigation using reinforcement learning in a PyBullet + OpenAI Gym environment.
- Two main RL algorithms are implemented: Proximal Policy Optimization (PPO) and Trust Region Policy Optimization (TRPO).
- The codebase supports both single-drone and multi-drone (formation) scenarios.

## Key Components
- `drone_envs/` and `muti_formation/drone_envs/`: Environment definitions, configs, and 3D resources for simulation.
- `agent/` and `muti_formation/agent/`: RL agent implementations for PPO and TRPO, with separate logs and model checkpoints.
- `PPO_trainer.py`, `TRPO_trainer.py`, `muti_formation/PPO_multi_trainer.py`: Training entry points for single and multi-drone setups.
- `run.py`, `muti_formation/run_multi.py`: Scripts to run trained models and evaluate performance.
- Model and log files are stored in `agent/model/`, `agent/log/`, and their multi-drone equivalents.

## Developer Workflows
- **Training**: Run `python PPO_trainer.py` or `python TRPO_trainer.py` for single-drone; use `muti_formation/PPO_multi_trainer.py` for multi-drone.
- **Evaluation**: Use `python run.py --model PPO --version 0` (or similar) to test trained models and see real-time success rates.
- **Configuring GUI**: Modify `drone_envs/config.py` or `muti_formation/drone_envs/config.py` to change display settings (e.g., `"display": p.GUI`).
- **Results**: Training and evaluation logs are JSON files in `agent/log/` and `muti_formation/agent/log/`.

## Project-Specific Patterns
- **Environment Versioning**: Multiple environment versions (e.g., `drone_env_v0.py`, `drone_env_v1.py`, `drone_env_multi.py`) allow for incremental improvements and experiments.
- **Separation of Single/Multi-Drone**: The `muti_formation/` directory mirrors the main structure for multi-drone experiments, keeping code and data isolated.
- **Model/Log Naming**: Model and log files are versioned (e.g., `PPO_drone_v0.pth`, `PPO_multi_LOG.json`) for reproducibility.

## External Dependencies
- Requires PyBullet, OpenAI Gym, and standard RL libraries (see `requirements.txt` if present, or infer from imports).
- 3D models and URDFs are in `drone_envs/resources/` and `muti_formation/drone_envs/resources/`.

## Example: Training a PPO Agent
```sh
python PPO_trainer.py
# or for multi-drone
python muti_formation/PPO_multi_trainer.py
```

## Example: Evaluating a Model
```sh
python run.py --model PPO --version 0
```

## References
- See `README.md` for more usage details and links to relevant resources.
- For new environments or agents, follow the structure in `drone_envs/envs/` and `agent/`.
