"""Visualize IK (Jacobian pseudo-inverse) baseline rollouts.

Usage
-----
# Save a GIF (default)
python visualize_ik.py --obstacles 1 --episodes 3

# Interactive MuJoCo viewer  (Linux/Windows)
python visualize_ik.py --obstacles 1 --human

# Explicit output path / more obstacles
python visualize_ik.py --obstacles 3 --save-gif artifacts/results/rollout_ik_obs3.gif
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.baselines import ik_velocity_baseline
from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize IK baseline rollout.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--obstacles", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-gif", default="",
                        help="Output GIF path (auto-named if omitted)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--human", action="store_true",
                        help="Open interactive MuJoCo viewer instead of saving GIF")
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    args = parser.parse_args()

    if not args.human:
        gif_path = args.save_gif or f"artifacts/results/rollout_ik_obs{args.obstacles}.gif"
        print(f"GIF output: {gif_path}")

    render_mode = "human" if args.human else "rgb_array"
    env = ObstacleAvoidanceArmEnv(
        render_mode=render_mode,
        config=EnvConfig(obstacle_count=args.obstacles),
    )

    try:
        all_frames: list = []
        for ep in range(args.episodes):
            obs, _ = env.reset()
            if args.human:
                env.render()
                time.sleep(0.3)

            done = False
            truncated = False
            step = 0
            info = {}

            while not done and not truncated and step < args.max_steps:
                action = ik_velocity_baseline(env)
                obs, _, done, truncated, info = env.step(action)
                frame = env.render()
                if frame is not None:
                    all_frames.append(frame)
                if args.human:
                    time.sleep(1.0 / max(args.fps, 1))
                step += 1

            print(
                f"Episode {ep + 1}: "
                f"success={info.get('success', False)}  "
                f"collision={info.get('collision', False)}  "
                f"steps={step}  "
                f"distance={info.get('distance_to_target', -1):.4f}"
            )

        if args.human and args.hold_seconds > 0:
            print(f"Holding viewer for {args.hold_seconds:.1f}s ...")
            time.sleep(args.hold_seconds)

        if all_frames and not args.human:
            import imageio.v2 as imageio

            out = Path(gif_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(out, all_frames, fps=args.fps)
            print(f"Saved {len(all_frames)} frames → {out}")

    except RuntimeError as exc:
        msg = str(exc)
        if "launch_passive" in msg and "mjpython" in msg:
            raise SystemExit(
                "MuJoCo human rendering on macOS requires mjpython.\n"
                "Try: mjpython visualize_ik.py --human --obstacles 1\n"
                "Or save a GIF: python visualize_ik.py --obstacles 1"
            ) from exc
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
