from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and visualize the robot end-effector workspace.")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--output", default="artifacts/results/workspace_ur5e.png")
    args = parser.parse_args()

    env = ObstacleAvoidanceArmEnv(config=EnvConfig(obstacle_count=1))
    points = []
    for _ in range(args.samples):
        env.data.qpos[:] = env._sample_joint_configuration()
        env.data.qvel[:] = 0.0
        env.data.ctrl[:] = env.data.qpos[: env.model.nu]
        mujoco.mj_forward(env.model, env.data)
        points.append(env.data.site_xpos[env.ee_site_id].copy())
    env.close()

    pts = np.asarray(points)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, alpha=0.25, c=pts[:, 2], cmap="viridis")
    ax.set_title("UR5e End-Effector Workspace Samples")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1.0, 1.0, 0.7))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved workspace visualization to {output_path}")


if __name__ == "__main__":
    main()
