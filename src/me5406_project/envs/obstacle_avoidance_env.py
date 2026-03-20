from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
    control_dt: float = 0.05
    episode_steps: int = 200
    goal_tolerance: float = 0.05
    sensor_range: float = 0.5
    obstacle_count: int = 3
    action_scale: float = 1.0
    distance_reward_scale: float = 8.0
    time_penalty: float = 0.01
    action_penalty: float = 0.02
    collision_penalty: float = 25.0
    success_reward: float = 40.0
    obstacle_radius: float = 0.06


class ObstacleAvoidanceArmEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: str | None = None, config: EnvConfig | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.config = config or EnvConfig()

        asset_path = Path(__file__).resolve().parents[1] / "assets" / "arm_3dof.xml"
        self.model = mujoco.MjModel.from_xml_path(str(asset_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.obstacle_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
            for i in range(5)
        ]

        obs_dim = 3 + 3 + 3 + 1 + 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._prev_distance = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.np_random.uniform(low=-0.25, high=0.25, size=3)
        self.data.qvel[:] = 0.0
        self._place_target()
        self._place_obstacles()
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_distance = self._target_distance()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action * self.config.action_scale

        substeps = max(1, int(round(self.config.control_dt / self.model.opt.timestep)))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        distance = info["distance_to_target"]
        distance_reward = self.config.distance_reward_scale * (self._prev_distance - distance)
        self._prev_distance = distance

        reward = distance_reward
        reward -= self.config.time_penalty
        reward -= self.config.action_penalty * float(np.sum(np.square(action)))

        collision = info["collision"]
        success = distance <= self.config.goal_tolerance
        truncated = self._step_count >= self.config.episode_steps
        terminated = collision or success

        if collision:
            reward -= self.config.collision_penalty
        if success:
            reward += self.config.success_reward

        info["success"] = success
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer

                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, width=640, height=480)
            renderer.update_scene(self.data)
            return renderer.render()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[:3].copy()
        qvel = self.data.qvel[:3].copy()
        rel_target = self.data.mocap_pos[0] - self.data.site_xpos[self.ee_site_id]
        distance = np.array([np.linalg.norm(rel_target)], dtype=np.float64)
        sensors = self._local_obstacle_sensors()
        return np.concatenate([qpos, qvel, rel_target, distance, sensors]).astype(np.float32)

    def _get_info(self) -> dict:
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target = self.data.mocap_pos[0].copy()
        return {
            "ee_position": ee_pos,
            "target_position": target,
            "distance_to_target": float(np.linalg.norm(target - ee_pos)),
            "collision": self._has_collision(),
            "sensor_readings": self._local_obstacle_sensors(),
        }

    def _target_distance(self) -> float:
        return float(np.linalg.norm(self.data.mocap_pos[0] - self.data.site_xpos[self.ee_site_id]))

    def _place_target(self) -> None:
        x = self.np_random.uniform(0.35, 0.65)
        y = self.np_random.uniform(-0.25, 0.25)
        z = self.np_random.uniform(0.05, 0.35)
        self.data.mocap_pos[0] = np.array([x, y, z], dtype=np.float64)
        self.data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _place_obstacles(self) -> None:
        active = self.config.obstacle_count
        for i, body_id in enumerate(self.obstacle_body_ids):
            if i < active:
                pos = np.array(
                    [
                        self.np_random.uniform(0.18, 0.55),
                        self.np_random.uniform(-0.22, 0.22),
                        self.np_random.uniform(0.05, 0.32),
                    ],
                    dtype=np.float64,
                )
            else:
                pos = np.array([2.0 + i, 2.0, 2.0], dtype=np.float64)
            self.data.mocap_pos[1 + i] = pos
            self.data.mocap_quat[1 + i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _local_obstacle_sensors(self) -> np.ndarray:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        readings = []
        for i in range(3):
            obstacle_pos = self.data.mocap_pos[1 + i]
            distance = np.linalg.norm(obstacle_pos - ee_pos) - self.config.obstacle_radius
            normalized = np.clip(distance / self.config.sensor_range, 0.0, 1.0)
            readings.append(normalized)
        return np.asarray(readings, dtype=np.float64)

    def _has_collision(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name
            if geom1 is None or geom2 is None:
                continue
            arm_hit = geom1.startswith("link") or geom2.startswith("link")
            obstacle_hit = geom1.startswith("obstacle_") or geom2.startswith("obstacle_")
            if arm_hit and obstacle_hit:
                return True
        return False
