"""Visualize RRT* plan-then-execute rollouts.

Usage
-----
# Save a GIF (default)
python visualize_rrt_star.py --obstacles 1 --episodes 3

# Interactive MuJoCo viewer  (Linux/Windows)
python visualize_rrt_star.py --obstacles 1 --human

# Explicit output path
python visualize_rrt_star.py --obstacles 3 --save-gif artifacts/results/rollout_rrt_obs3.gif

The script prints per-episode metrics (success, collision, path length, planning
latency) and exits cleanly if no path is found for an episode.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from src.baselines import _smooth_path, rrt_star_plan
from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


def run_episode(
    env: ObstacleAvoidanceArmEnv,
    max_iterations: int,
    seed: int,
    human: bool,
    fps: int,
) -> tuple[list, dict]:
    """Plan with RRT*, execute, and collect render frames.

    Returns (frames, info_dict).  frames is empty when human=True.
    """
    rng = np.random.default_rng(seed)
    env.reset(seed=int(rng.integers(0, 2**31)))

    if human:
        env.render()
        time.sleep(0.3)

    # ── Planning (mutates qpos/qvel freely) ──────────────────────────────────
    qpos_saved = env.data.qpos.copy()
    qvel_saved = env.data.qvel.copy()

    plan_start = time.perf_counter()
    path = rrt_star_plan(env, max_iterations=max_iterations, rng=rng)
    if path is not None:
        path = _smooth_path(env, path)
    plan_time = time.perf_counter() - plan_start

    # Restore env to post-reset state for execution
    env.data.qpos[:] = qpos_saved
    env.data.qvel[:] = qvel_saved
    mujoco.mj_forward(env.model, env.data)

    if path is None:
        return [], {
            "success": False,
            "collision": False,
            "path_found": False,
            "path_length": 0,
            "planning_latency_sec": plan_time,
            "steps": 0,
        }

    # ── Execution ─────────────────────────────────────────────────────────────
    frames: list = []
    total_steps = 0

    for wp in path[1:]:
        for _ in range(40):                         # max steps per waypoint
            q_curr = env.data.qpos[:3].copy()
            if float(np.linalg.norm(wp - q_curr)) < 0.08:
                break
            action = np.clip(4.0 * (wp - q_curr), -1.0, 1.0).astype(np.float32)
            _, _, done, truncated, info = env.step(action)
            total_steps += 1

            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if human:
                time.sleep(1.0 / max(fps, 1))

            if done or truncated:
                return frames, {
                    "success": bool(info.get("success", False)),
                    "collision": bool(info.get("collision", False)),
                    "path_found": True,
                    "path_length": len(path),
                    "planning_latency_sec": plan_time,
                    "steps": total_steps,
                }

    # Final probe step to read terminal info
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    total_steps += 1
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    return frames, {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "path_found": True,
        "path_length": len(path),
        "planning_latency_sec": plan_time,
        "steps": total_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RRT* plan-then-execute rollout.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--obstacles", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3000,
                        help="RRT* tree expansion budget per episode")
    parser.add_argument("--save-gif", default="",
                        help="Output GIF path (auto-named if omitted)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--human", action="store_true",
                        help="Open interactive MuJoCo viewer instead of saving GIF")
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.human:
        gif_path = args.save_gif or f"artifacts/results/rollout_rrt_obs{args.obstacles}.gif"
        print(f"GIF output: {gif_path}")

    render_mode = "human" if args.human else "rgb_array"
    env = ObstacleAvoidanceArmEnv(
        render_mode=render_mode,
        config=EnvConfig(obstacle_count=args.obstacles),
    )

    try:
        all_frames: list = []
        for ep in range(args.episodes):
            frames, info = run_episode(
                env,
                max_iterations=args.max_iterations,
                seed=args.seed + ep,
                human=args.human,
                fps=args.fps,
            )
            all_frames.extend(frames)

            status = "NOT FOUND" if not info["path_found"] else (
                f"length={info['path_length']}"
            )
            print(
                f"Episode {ep + 1}: "
                f"path={status}  "
                f"plan_time={info['planning_latency_sec']:.2f}s  "
                f"success={info['success']}  "
                f"collision={info['collision']}  "
                f"steps={info['steps']}"
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
                "Try: mjpython visualize_rrt_star.py --human --obstacles 1\n"
                "Or save a GIF: python visualize_rrt_star.py --obstacles 1"
            ) from exc
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
