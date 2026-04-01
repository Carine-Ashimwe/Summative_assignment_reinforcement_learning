from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from environment.custom_env import MaternalChildHealthMissionEnv, MissionConfig


sns.set_theme(style="whitegrid")


def build_env(seed: int = 7) -> gym.Env:
    env = MaternalChildHealthMissionEnv(config=MissionConfig(), render_mode=None, seed=seed)
    return Monitor(env)


def build_vec_env(seed: int = 7) -> VecMonitor:
    env = DummyVecEnv([lambda: MaternalChildHealthMissionEnv(config=MissionConfig(), render_mode=None, seed=seed)])
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


def reinforce_sweep() -> List[Dict]:
    return [
        {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64, "ent_coef": 0.00},
        {"learning_rate": 5e-4, "gamma": 0.99, "hidden_size": 128, "ent_coef": 0.005},
        {"learning_rate": 3e-4, "gamma": 0.97, "hidden_size": 128, "ent_coef": 0.01},
        {"learning_rate": 8e-4, "gamma": 0.98, "hidden_size": 256, "ent_coef": 0.00},
        {"learning_rate": 2e-4, "gamma": 0.995, "hidden_size": 256, "ent_coef": 0.02},
        {"learning_rate": 6e-4, "gamma": 0.96, "hidden_size": 64, "ent_coef": 0.015},
        {"learning_rate": 1e-3, "gamma": 0.985, "hidden_size": 512, "ent_coef": 0.003},
        {"learning_rate": 4e-4, "gamma": 0.99, "hidden_size": 256, "ent_coef": 0.008},
        {"learning_rate": 7e-4, "gamma": 0.97, "hidden_size": 128, "ent_coef": 0.01},
        {"learning_rate": 3e-4, "gamma": 0.995, "hidden_size": 512, "ent_coef": 0.005},
    ]


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_reinforce(policy: PolicyNet, env: gym.Env, episodes: int = 5) -> Dict[str, float]:
    rewards = []
    lengths = []
    adverse_events = []
    device = next(policy.parameters()).device

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        last_info = {}

        while not done and not truncated:
            with torch.no_grad():
                logits = policy(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            last_info = info

        rewards.append(ep_reward)
        lengths.append(steps)
        adverse_events.append(last_info.get("preventable_adverse_events", 0))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_adverse_events": float(np.mean(adverse_events)),
    }


def train_reinforce(run_id: str, params: Dict, timesteps: int, seed: int, eval_episodes: int = 5) -> Dict:
    env = build_env(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=int(params["hidden_size"]))
    optimizer = optim.Adam(policy.parameters(), lr=float(params["learning_rate"]))

    gamma = float(params["gamma"])
    ent_coef = float(params["ent_coef"])
    total_steps = 0

    while total_steps < timesteps:
        obs, _ = env.reset()
        done = False
        truncated = False
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        rewards: List[float] = []

        while not done and not truncated:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            obs, reward, done, truncated, _ = env.step(int(action.item()))
            rewards.append(float(reward))
            total_steps += 1
            if total_steps >= timesteps:
                break

        returns: List[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.append(g)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if returns_t.numel() > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        policy_loss = torch.tensor(0.0)
        entropy_bonus = torch.tensor(0.0)
        for lp, ret, ent in zip(log_probs, returns_t, entropies):
            policy_loss = policy_loss - lp * ret
            entropy_bonus = entropy_bonus + ent

        loss = policy_loss - ent_coef * entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    model_path = Path("models/pg") / f"{run_id}.pt"
    ensure_dir(model_path.parent)
    torch.save(policy.state_dict(), model_path)

    eval_env = build_env(seed=seed + 100)
    metrics = evaluate_reinforce(policy, eval_env, episodes=eval_episodes)
    payload = {
        "run_id": run_id,
        "algorithm": "REINFORCE",
        "timesteps": timesteps,
        "eval_episodes": eval_episodes,
        **params,
        **metrics,
    }
    save_json(Path("results") / "reinforce" / f"{run_id}.json", payload)

    env.close()
    eval_env.close()
    return payload


def train_ppo(run_id: str, params: Dict, timesteps: int, seed: int, eval_episodes: int = 5) -> Dict:
    vec_env = build_vec_env(seed=seed)
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

    eval_env = build_env(seed=seed + 100)
    metrics = evaluate_policy_simple(model, eval_env, episodes=eval_episodes)
    payload = {
        "run_id": run_id,
        "algorithm": "PPO",
        "timesteps": timesteps,
        "eval_episodes": eval_episodes,
        **params,
        **metrics,
    }
    save_json(Path("results") / "ppo" / f"{run_id}.json", payload)

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


def plot_results() -> None:
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _read(path: str) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)

    dqn = _read("results/dqn_experiments.csv")
    ppo = _read("results/ppo_experiments.csv")
    reinf = _read("results/reinforce_experiments.csv")

    # cumulative reward curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, frame) in zip(axes.ravel(), [("DQN", dqn), ("PPO", ppo), ("REINFORCE", reinf)]):
        if frame.empty:
            ax.set_title(f"{name} (no data)")
            continue
        frame = frame.copy()
        frame["run_order"] = range(1, len(frame) + 1)
        sns.lineplot(data=frame, x="run_order", y="mean_reward", marker="o", ax=ax)
        ax.set_title(f"{name} cumulative reward")
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_reward_curves.png", dpi=180)
    plt.close(fig)

    # dqn objective curves
    if not dqn.empty:
        dqn2 = dqn.copy()
        dqn2["run_order"] = range(1, len(dqn2) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.scatterplot(data=dqn2, x="buffer_size", y="mean_reward", hue="gamma", size="batch_size", ax=axes[0])
        axes[0].set_title("DQN objective sensitivity")
        sns.lineplot(data=dqn2, x="run_order", y="std_reward", marker="o", ax=axes[1])
        axes[1].set_title("DQN stability")
        fig.tight_layout()
        fig.savefig(out_dir / "dqn_objective_curves.png", dpi=180)
        plt.close(fig)

    # pg entropy curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, frame) in zip(axes, [("PPO", ppo), ("REINFORCE", reinf)]):
        if frame.empty or "ent_coef" not in frame.columns:
            ax.set_title(f"{name} entropy (no data)")
            continue
        sns.scatterplot(data=frame, x="ent_coef", y="mean_reward", hue="gamma", ax=ax)
        ax.set_title(f"{name} entropy vs reward")
    fig.tight_layout()
    fig.savefig(out_dir / "pg_entropy_curves.png", dpi=180)
    plt.close(fig)

    # convergence/generalization
    rows = []
    for name, frame in [("DQN", dqn), ("PPO", ppo), ("REINFORCE", reinf)]:
        if frame.empty:
            continue
        rows.append({
            "algorithm": name,
            "best_mean_reward": frame["mean_reward"].max(),
            "avg_adverse_events": frame["mean_adverse_events"].mean(),
        })
    if rows:
        summary = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=summary, x="algorithm", y="best_mean_reward", ax=axes[0])
        axes[0].set_title("Best reward by algorithm")
        sns.barplot(data=summary, x="algorithm", y="avg_adverse_events", ax=axes[1])
        axes[1].set_title("Generalization: adverse events")
        fig.tight_layout()
        fig.savefig(out_dir / "convergence_and_generalization.png", dpi=180)
        plt.close(fig)

    print(f"Saved analysis plots to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO or REINFORCE on maternal-child health mission environment")
    parser.add_argument("--algo", choices=["ppo", "reinforce"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=140_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--run-id", type=str, default="pg_manual")
    parser.add_argument("--max-runs", type=int, default=10, help="Number of configs to run during sweep")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Fast debug mode: fewer timesteps and fewer sweep runs")
    parser.add_argument("--plot-results", action="store_true", help="Generate required analysis plots from results CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.plot_results:
        plot_results()
        return

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
        sweep = reinforce_sweep()
        trainer = train_reinforce
        csv_path = Path("results") / "reinforce_experiments.csv"

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
        eval_episodes=eval_episodes,
    )
    append_csv(csv_path, [payload])


if __name__ == "__main__":
    main()
