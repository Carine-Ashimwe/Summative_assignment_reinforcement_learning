from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from environment.custom_env import MaternalChildHealthMissionEnv, MissionConfig


def build_env_with_mode(seed: int = 7) -> gym.Env:
    env = MaternalChildHealthMissionEnv(
        config=MissionConfig(),
        render_mode=None,
        seed=seed,
    )
    return Monitor(env)


def build_vec_env(seed: int = 7) -> VecMonitor:
    env = DummyVecEnv(
        [
            lambda: MaternalChildHealthMissionEnv(
                config=MissionConfig(),
                render_mode=None,
                seed=seed,
            )
        ]
    )
    return VecMonitor(env)


def evaluate_policy_simple(model, env: gym.Env, episodes: int = 5) -> Dict[str, float]:
    rewards = []
    lengths = []
    adverse_events = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        steps = 0
        last_info = {}

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            episode_reward += reward
            steps += 1
            last_info = info

        rewards.append(episode_reward)
        lengths.append(steps)
        adverse_events.append(last_info.get("preventable_adverse_events", 0))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_adverse_events": float(np.mean(adverse_events)),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def default_sweep() -> List[Dict]:
    return [
        {"learning_rate": 1e-4, "gamma": 0.97, "buffer_size": 30_000, "batch_size": 64, "target_update_interval": 300, "exploration_fraction": 0.2},
        {"learning_rate": 3e-4, "gamma": 0.97, "buffer_size": 50_000, "batch_size": 64, "target_update_interval": 300, "exploration_fraction": 0.2},
        {"learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 128, "target_update_interval": 500, "exploration_fraction": 0.3},
        {"learning_rate": 5e-4, "gamma": 0.95, "buffer_size": 20_000, "batch_size": 64, "target_update_interval": 200, "exploration_fraction": 0.15},
        {"learning_rate": 2e-4, "gamma": 0.99, "buffer_size": 80_000, "batch_size": 128, "target_update_interval": 700, "exploration_fraction": 0.35},
        {"learning_rate": 7e-5, "gamma": 0.98, "buffer_size": 70_000, "batch_size": 256, "target_update_interval": 1000, "exploration_fraction": 0.2},
        {"learning_rate": 4e-4, "gamma": 0.96, "buffer_size": 40_000, "batch_size": 64, "target_update_interval": 300, "exploration_fraction": 0.1},
        {"learning_rate": 1e-4, "gamma": 0.995, "buffer_size": 100_000, "batch_size": 256, "target_update_interval": 1200, "exploration_fraction": 0.4},
        {"learning_rate": 2e-4, "gamma": 0.985, "buffer_size": 60_000, "batch_size": 128, "target_update_interval": 600, "exploration_fraction": 0.25},
        {"learning_rate": 8e-5, "gamma": 0.99, "buffer_size": 120_000, "batch_size": 256, "target_update_interval": 1500, "exploration_fraction": 0.5},
    ]


def train_one(run_id: str, params: Dict, timesteps: int, seed: int, eval_episodes: int = 5) -> Dict:
    vec_env = build_vec_env(seed=seed)
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        target_update_interval=params["target_update_interval"],
        exploration_fraction=params["exploration_fraction"],
        learning_starts=2000,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="results/tb/dqn",
        seed=seed,
    )
    model.learn(total_timesteps=timesteps)

    model_path = Path("models/dqn") / f"{run_id}.zip"
    ensure_dir(model_path.parent)
    model.save(model_path)

    eval_env = build_env_with_mode(seed=seed + 100)
    metrics = evaluate_policy_simple(model, eval_env, episodes=eval_episodes)

    payload = {
        "run_id": run_id,
        "algorithm": "DQN",
        "timesteps": timesteps,
        "eval_episodes": eval_episodes,
        **params,
        **metrics,
    }
    save_json(Path("results") / "dqn" / f"{run_id}.json", payload)
    eval_env.close()
    vec_env.close()
    return payload


def append_csv(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_sweep(timesteps: int, seed: int, max_runs: int = 10, eval_episodes: int = 5) -> None:
    rows = []
    for idx, params in enumerate(default_sweep()[:max_runs], start=1):
        run_id = f"dqn_run_{idx:02d}"
        row = train_one(
            run_id=run_id,
            params=params,
            timesteps=timesteps,
            seed=seed + idx,
            eval_episodes=eval_episodes,
        )
        rows.append(row)
    append_csv(Path("results") / "dqn_experiments.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on maternal-child health mission environment")
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sweep", action="store_true", help="Run all 10 hyperparameter experiments")
    parser.add_argument("--run-id", type=str, default="dqn_manual")
    parser.add_argument("--max-runs", type=int, default=10, help="Number of configs to run during sweep")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Fast debug mode: fewer timesteps and fewer sweep runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timesteps = args.timesteps
    max_runs = args.max_runs
    eval_episodes = args.eval_episodes

    if args.quick:
        timesteps = min(timesteps, 20_000)
        max_runs = min(max_runs, 3)
        eval_episodes = min(eval_episodes, 2)

    if args.sweep:
        run_sweep(
            timesteps=timesteps,
            seed=args.seed,
            max_runs=max_runs,
            eval_episodes=eval_episodes,
        )
        return

    params = default_sweep()[0]
    payload = train_one(
        run_id=args.run_id,
        params=params,
        timesteps=timesteps,
        seed=args.seed,
        eval_episodes=eval_episodes,
    )
    append_csv(Path("results") / "dqn_experiments.csv", [payload])


if __name__ == "__main__":
    main()
