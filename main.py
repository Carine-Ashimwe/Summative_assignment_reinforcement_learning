from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from environment.custom_env import MaternalChildHealthMissionEnv, MissionConfig
from training.pg_training import PolicyNet


def load_agent(algo: str, model_path: Path, env: MaternalChildHealthMissionEnv):
    algo = algo.lower()
    if algo == "dqn":
        model = DQN.load(str(model_path))
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    if algo == "ppo":
        model = PPO.load(str(model_path))
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    if algo == "reinforce":
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        state_dict = torch.load(model_path, map_location="cpu")
        hidden_size = int(state_dict["net.0.weight"].shape[0])
        policy = PolicyNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=hidden_size)
        policy.load_state_dict(state_dict)
        policy.eval()

        def _predict(obs):
            with torch.no_grad():
                logits = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(logits, dim=-1).item())

        return _predict

    raise ValueError(f"Unsupported algo: {algo}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained maternal-child health mission agent")
    parser.add_argument("--algo", choices=["dqn", "ppo", "reinforce"], default="ppo")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model file")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=4, help="GUI playback FPS (lower is slower)")
    parser.add_argument("--step-delay", type=float, default=0.0, help="Extra delay per environment step in seconds")
    parser.add_argument("--random-demo", action="store_true", help="Run random actions and save static artifacts")
    parser.add_argument("--steps", type=int, default=180, help="Number of random steps for --random-demo")
    return parser.parse_args()


def run_random_demo(args: argparse.Namespace) -> None:
    env = MaternalChildHealthMissionEnv(config=MissionConfig(), render_mode="rgb_array", seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    frames = []
    trace = []

    for step in range(args.steps):
        action = int(env.action_space.sample())
        obs, reward, done, truncated, info = env.step(action)
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame)
        trace.append(
            {
                "step": step + 1,
                "action": action,
                "action_name": info.get("action_name", "unknown"),
                "reward": float(reward),
                "total_reward": float(info.get("total_reward", 0.0)),
                "maternal_risk": float(env.state[0]),
                "neonatal_risk": float(env.state[1]),
                "budget_remaining": float(env.state[9]),
            }
        )
        if done or truncated:
            break

    env.close()
    out_dir = Path("results/random_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "random_agent_demo.gif"
    json_path = out_dir / "random_actions_trace.json"
    if frames:
        imageio.mimsave(gif_path, frames, fps=max(1, args.fps))
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    print(f"Saved random demo GIF: {gif_path}")
    print(f"Saved random trace JSON: {json_path}")


def main() -> None:
    args = parse_args()
    if args.random_demo:
        run_random_demo(args)
        return

    if not args.model:
        raise ValueError("--model is required unless --random-demo is used")

    env = MaternalChildHealthMissionEnv(
        config=MissionConfig(),
        render_mode="human",
        seed=args.seed,
    )
    env.set_render_fps(args.fps)

    model_path = Path(args.model)
    act = load_agent(args.algo, model_path, env)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0

        print(f"\n=== Episode {ep + 1} ===")
        while not done and not truncated:
            action = act(obs)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            env.render()
            if args.step_delay > 0:
                time.sleep(args.step_delay)
            print(
                f"Day={info['day']:03d} | Action={info['action_name']:<38} | "
                f"Reward={reward:>6.3f} | Total={info['total_reward']:>8.3f} | "
                f"AdverseEvents={info['preventable_adverse_events']}"
            )

        print(f"Episode reward: {ep_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
