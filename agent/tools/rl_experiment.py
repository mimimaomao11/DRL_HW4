"""Tool 2: RL Experiment Runner — trains an SB3 agent and persists results to experiment_db.json."""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "memory" / "experiment_db.json"

SUPPORTED_ALGORITHMS = ["DQN", "PPO", "SAC"]


def _load_db() -> dict:
    if DB_PATH.exists():
        with open(DB_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_db(db: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def run_rl_experiment(
    algorithm: str,
    environment: str,
    n_steps: int,
    hyperparams: dict | None = None,
) -> dict:
    """Run an RL training experiment using Stable-Baselines3.

    Args:
        algorithm: One of "DQN", "PPO", "SAC"
        environment: Gymnasium environment ID (e.g. "CartPole-v1")
        n_steps: Total training timesteps
        hyperparams: Optional dict of SB3 algorithm kwargs

    Returns:
        dict with experiment_id, algorithm, environment, mean_reward, std_reward, status
    """
    exp_id = str(uuid.uuid4())[:8]
    hyperparams = hyperparams or {}

    if algorithm not in SUPPORTED_ALGORITHMS:
        return {
            "experiment_id": exp_id,
            "error": f"Unsupported algorithm '{algorithm}'. Choose from {SUPPORTED_ALGORITHMS}.",
            "status": "failed",
        }

    try:
        import gymnasium as gym
        from stable_baselines3 import DQN, PPO, SAC
        from stable_baselines3.common.evaluation import evaluate_policy

        algo_map = {"DQN": DQN, "PPO": PPO, "SAC": SAC}
        AlgoClass = algo_map[algorithm]

        env = gym.make(environment)
        model = AlgoClass("MlpPolicy", env, verbose=0, **hyperparams)
        model.learn(total_timesteps=n_steps)

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, warn=False)
        env.close()

        result = {
            "experiment_id": exp_id,
            "algorithm": algorithm,
            "environment": environment,
            "n_steps": n_steps,
            "hyperparams": hyperparams,
            "mean_reward": round(float(mean_reward), 2),
            "std_reward": round(float(std_reward), 2),
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
        }

    except ImportError as e:
        result = {
            "experiment_id": exp_id,
            "error": f"Missing dependency: {e}. Install with: pip install stable-baselines3 gymnasium",
            "status": "failed",
        }
    except Exception as e:
        result = {
            "experiment_id": exp_id,
            "algorithm": algorithm,
            "environment": environment,
            "error": str(e),
            "status": "failed",
        }

    db = _load_db()
    db[exp_id] = result
    _save_db(db)

    return result
