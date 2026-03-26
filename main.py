from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import MaternalChildHealthMissionEnv, MissionConfig


def load_agent(algo: str, model_path: Path, env: MaternalChildHealthMissionEnv):
    algo = algo.lower()
    if algo == "dqn":
        model = DQN.load(str(model_path))
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    if algo == "ppo":
        model = PPO.load(str(model_path))
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    if algo == "a2c":
        model = A2C.load(str(model_path))
        return lambda obs: int(model.predict(obs, deterministic=True)[0])

    raise ValueError(f"Unsupported algo: {algo}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained maternal-child health mission agent")
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c"], default="ppo")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model file")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=4, help="GUI playback FPS (lower is slower)")
    parser.add_argument("--step-delay", type=float, default=0.0, help="Extra delay per environment step in seconds")
    parser.add_argument("--focused", action="store_true", help="Use focused maternal-neonatal (6-feature) observation mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = MaternalChildHealthMissionEnv(
        config=MissionConfig(focused_mnh_mode=args.focused),
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
