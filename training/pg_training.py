from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from environment.custom_env import MaternalChildHealthMissionEnv, MissionConfig


def build_env_with_mode(seed: int = 7, focused_mnh_mode: bool = False) -> gym.Env:
    env = MaternalChildHealthMissionEnv(
        config=MissionConfig(focused_mnh_mode=focused_mnh_mode),
        render_mode=None,
        seed=seed,
    )
    return Monitor(env)


def build_vec_env(seed: int = 7, focused_mnh_mode: bool = False) -> VecMonitor:
    env = DummyVecEnv(
        [
            lambda: MaternalChildHealthMissionEnv(
                config=MissionConfig(focused_mnh_mode=focused_mnh_mode),
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


def ppo_sweep() -> List[Dict]:
    return [
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 128, "ent_coef": 0.01, "clip_range": 0.2},
        {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 256, "ent_coef": 0.005, "clip_range": 0.2},
        {"learning_rate": 5e-4, "gamma": 0.97, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.02, "clip_range": 0.15},
        {"learning_rate": 2e-4, "gamma": 0.995, "n_steps": 2048, "batch_size": 128, "ent_coef": 0.001, "clip_range": 0.25},
        {"learning_rate": 3e-4, "gamma": 0.98, "n_steps": 512, "batch_size": 64, "ent_coef": 0.03, "clip_range": 0.2},
        {"learning_rate": 8e-5, "gamma": 0.99, "n_steps": 4096, "batch_size": 256, "ent_coef": 0.0, "clip_range": 0.2},
        {"learning_rate": 2e-4, "gamma": 0.96, "n_steps": 1024, "batch_size": 128, "ent_coef": 0.015, "clip_range": 0.3},
        {"learning_rate": 4e-4, "gamma": 0.985, "n_steps": 2048, "batch_size": 256, "ent_coef": 0.008, "clip_range": 0.18},
        {"learning_rate": 1.5e-4, "gamma": 0.99, "n_steps": 1536, "batch_size": 128, "ent_coef": 0.01, "clip_range": 0.22},
        {"learning_rate": 6e-5, "gamma": 0.995, "n_steps": 4096, "batch_size": 512, "ent_coef": 0.003, "clip_range": 0.2},
    ]


def a2c_sweep() -> List[Dict]:
    return [
        {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 1.0},
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.005, "vf_coef": 0.5, "gae_lambda": 0.98},
        {"learning_rate": 1e-3, "gamma": 0.97, "n_steps": 5, "ent_coef": 0.02, "vf_coef": 0.4, "gae_lambda": 0.95},
        {"learning_rate": 2e-4, "gamma": 0.995, "n_steps": 20, "ent_coef": 0.0, "vf_coef": 0.7, "gae_lambda": 1.0},
        {"learning_rate": 6e-4, "gamma": 0.98, "n_steps": 8, "ent_coef": 0.015, "vf_coef": 0.6, "gae_lambda": 0.97},
        {"learning_rate": 4e-4, "gamma": 0.96, "n_steps": 12, "ent_coef": 0.03, "vf_coef": 0.5, "gae_lambda": 0.95},
        {"learning_rate": 8e-4, "gamma": 0.99, "n_steps": 16, "ent_coef": 0.008, "vf_coef": 0.4, "gae_lambda": 1.0},
        {"learning_rate": 1e-4, "gamma": 0.995, "n_steps": 20, "ent_coef": 0.001, "vf_coef": 0.8, "gae_lambda": 0.99},
        {"learning_rate": 5e-4, "gamma": 0.985, "n_steps": 10, "ent_coef": 0.02, "vf_coef": 0.6, "gae_lambda": 0.96},
        {"learning_rate": 2.5e-4, "gamma": 0.99, "n_steps": 15, "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.97},
    ]


def train_ppo(run_id: str, params: Dict, timesteps: int, seed: int, focused: bool = False, eval_episodes: int = 5) -> Dict:
    vec_env = build_vec_env(seed=seed, focused_mnh_mode=focused)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        ent_coef=params["ent_coef"],
        clip_range=params["clip_range"],
        verbose=1,
        tensorboard_log="results/tb/ppo",
        seed=seed,
    )
    model.learn(total_timesteps=timesteps)

    model_path = Path("models/pg") / f"{run_id}.zip"
    ensure_dir(model_path.parent)
    model.save(model_path)

    eval_env = build_env_with_mode(seed=seed + 100, focused_mnh_mode=focused)
    metrics = evaluate_policy_simple(model, eval_env, episodes=eval_episodes)
    payload = {
        "run_id": run_id,
        "algorithm": "PPO",
        "focused_mnh_mode": focused,
        "timesteps": timesteps,
        "eval_episodes": eval_episodes,
        **params,
        **metrics,
    }
    save_json(Path("results") / "ppo" / f"{run_id}.json", payload)

    eval_env.close()
    vec_env.close()
    return payload


def train_a2c(run_id: str, params: Dict, timesteps: int, seed: int, focused: bool = False, eval_episodes: int = 5) -> Dict:
    vec_env = build_vec_env(seed=seed, focused_mnh_mode=focused)
    model = A2C(
        "MlpPolicy",
        vec_env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        n_steps=params["n_steps"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        gae_lambda=params["gae_lambda"],
        verbose=1,
        tensorboard_log="results/tb/a2c",
        seed=seed,
    )
    model.learn(total_timesteps=timesteps)

    model_path = Path("models/pg") / f"{run_id}.zip"
    ensure_dir(model_path.parent)
    model.save(model_path)

    eval_env = build_env_with_mode(seed=seed + 100, focused_mnh_mode=focused)
    metrics = evaluate_policy_simple(model, eval_env, episodes=eval_episodes)
    payload = {
        "run_id": run_id,
        "algorithm": "A2C",
        "focused_mnh_mode": focused,
        "timesteps": timesteps,
        "eval_episodes": eval_episodes,
        **params,
        **metrics,
    }
    save_json(Path("results") / "a2c" / f"{run_id}.json", payload)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO or A2C on maternal-child health mission environment")
    parser.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=140_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--run-id", type=str, default="pg_manual")
    parser.add_argument("--focused", action="store_true", help="Use focused maternal-neonatal (6-feature) observation mode")
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

    if args.algo == "ppo":
        sweep = ppo_sweep()
        trainer = train_ppo
        csv_path = Path("results") / "ppo_experiments.csv"
    else:
        sweep = a2c_sweep()
        trainer = train_a2c
        csv_path = Path("results") / "a2c_experiments.csv"

    if args.sweep:
        rows = []
        for idx, params in enumerate(sweep[:max_runs], start=1):
            run_id = f"{args.algo}_run_{idx:02d}"
            rows.append(
                trainer(
                    run_id=run_id,
                    params=params,
                    timesteps=timesteps,
                    seed=args.seed + idx,
                    focused=args.focused,
                    eval_episodes=eval_episodes,
                )
            )
        append_csv(csv_path, rows)
        return

    payload = trainer(
        run_id=args.run_id,
        params=sweep[0],
        timesteps=timesteps,
        seed=args.seed,
        focused=args.focused,
        eval_episodes=eval_episodes,
    )
    append_csv(csv_path, [payload])


if __name__ == "__main__":
    main()
