from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a standalone robot model snapshot.")
    parser.add_argument("--output", default="artifacts/results/model_snapshot_ur5e.png")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    asset_path = Path(__file__).resolve().parents[1] / "src" / "assets" / "ur5e_obstacle_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(asset_path))
    data = mujoco.MjData(model)

    # A slightly lifted pose that separates shoulder, elbow, forearm, and wrist visually.
    data.qpos[:] = np.array([-1.25, -1.55, 1.45, -1.35, -1.35, 0.35], dtype=np.float64)
    data.ctrl[:] = data.qpos[: model.nu]
    data.qvel[:] = 0.0
    data.mocap_pos[0] = np.array([0.48, -0.18, 0.42], dtype=np.float64)
    data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for i in range(5):
        data.mocap_pos[1 + i] = np.array([2.0 + i, 2.0, 2.0], dtype=np.float64)
        data.mocap_quat[1 + i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    camera = mujoco.MjvCamera()
    camera.lookat = np.array([0.30, -0.05, 0.38], dtype=np.float64)
    camera.distance = 1.45
    camera.azimuth = 132
    camera.elevation = -18

    renderer.update_scene(data, camera=camera)
    image = renderer.render()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, image)
    print(output_path)


if __name__ == "__main__":
    main()
