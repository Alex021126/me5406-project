from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from stable_baselines3 import SAC

from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv
from src.evaluation import model_uses_goal_conditioning


def resolve_model_path(model_path: str | None) -> str:
    if model_path:
        return model_path

    candidates = sorted(Path("artifacts/models").glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            "No model_path was provided and no trained model was found in artifacts/models/. "
            "Train a model first or pass the .zip path explicitly."
        )
    return str(candidates[-1])


def resolve_gif_path(save_gif: str, obstacle_count: int) -> str:
    if save_gif:
        return save_gif
    return f"artifacts/results/rollout_obs{obstacle_count}.gif"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a trained SAC policy rollout.")
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--obstacles", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-gif", default="")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)
    save_gif = "" if args.human else resolve_gif_path(args.save_gif, args.obstacles)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {model_path}")
    if save_gif:
        print(f"GIF output: {save_gif}")

    goal_conditioned = model_uses_goal_conditioning(model_path)
    render_mode = "human" if args.human else "rgb_array"
    env = ObstacleAvoidanceArmEnv(
        render_mode=render_mode,
        config=EnvConfig(obstacle_count=args.obstacles),
        goal_conditioned=goal_conditioned,
    )
    model = SAC.load(model_path, env=env, device=device)

    try:
        frames = []
        for episode in range(args.episodes):
            obs, _ = env.reset()
            if args.human:
                env.render()
                time.sleep(0.5)
            done = False
            truncated = False
            step = 0
            info = {}

            while not done and not truncated and step < args.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(action)
                frame = env.render()
                if frame is not None and save_gif:
                    frames.append(frame)
                if args.human:
                    time.sleep(1.0 / max(args.fps, 1))
                step += 1

            print(
                f"Episode {episode + 1}: success={info.get('success', False)} "
                f"collision={info.get('collision', False)} "
                f"steps={step} distance={info.get('distance_to_target', -1):.4f}"
            )

        if args.human and args.hold_seconds > 0:
            print(f"Holding viewer open for {args.hold_seconds:.1f}s")
            time.sleep(args.hold_seconds)

        if save_gif:
            import imageio.v2 as imageio

            output_path = Path(save_gif)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(output_path, frames, fps=args.fps)
            print(f"Saved GIF to {output_path}")
    except RuntimeError as exc:
        message = str(exc)
        if "launch_passive" in message and "mjpython" in message:
            raise SystemExit(
                "MuJoCo human rendering on macOS must be launched with `mjpython`.\n"
                "Try: `mjpython visualize.py --human --obstacles 3`\n"
                "Or export a GIF with: `python visualize.py --save-gif artifacts/results/rollout_obs3.gif`."
            ) from exc
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
