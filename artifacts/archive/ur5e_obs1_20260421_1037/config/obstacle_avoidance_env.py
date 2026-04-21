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
    episode_steps: int = 150
    goal_tolerance: float = 0.07
    sensor_range: float = 0.65
    obstacle_count: int = 3
    action_scale: float = 1.0
    distance_reward_scale: float = 10.0
    time_penalty: float = 0.01
    action_penalty: float = 0.02
    action_delta_penalty: float = 0.0
    clearance_penalty_scale: float = 2.0
    clearance_threshold: float = 0.12
    collision_penalty: float = 35.0
    success_reward: float = 100.0
    obstacle_radius: float = 0.05
    max_reset_tries: int = 80
    min_target_ee_distance: float = 0.15
    min_target_base_distance: float = 0.38
    min_obstacle_ee_distance: float = 0.18
    min_obstacle_target_distance: float = 0.18
    min_obstacle_spacing: float = 0.22
    reachable_target_samples: int = 120
    target_x_bounds: tuple[float, float] = (-0.36, 0.18)
    target_y_bounds: tuple[float, float] = (0.26, 0.62)
    target_z_bounds: tuple[float, float] = (0.28, 0.66)
    obstacle_x_bounds: tuple[float, float] = (-0.32, 0.16)
    obstacle_y_bounds: tuple[float, float] = (0.24, 0.60)
    obstacle_z_bounds: tuple[float, float] = (0.26, 0.62)


class ObstacleAvoidanceArmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        config: EnvConfig | None = None,
        goal_conditioned: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.config = config or EnvConfig()
        self.goal_conditioned = goal_conditioned

        asset_path = Path(__file__).resolve().parents[1] / "assets" / "ur5e_obstacle_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(asset_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.renderer = None

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.obstacle_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
            for i in range(5)
        ]
        self._joint_count = self.model.nu
        self._joint_lower_bounds = self.model.jnt_range[: self._joint_count, 0].copy()
        self._joint_upper_bounds = self.model.jnt_range[: self._joint_count, 1].copy()
        self._home_qpos = self.model.key_qpos[0].copy() if self.model.nkey else np.zeros(self._joint_count)
        self._reset_qpos_jitter = np.array([0.75, 0.45, 0.55, 0.55, 0.55, 0.75], dtype=np.float64)
        self._ctrl_lower_bounds = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_upper_bounds = self.model.actuator_ctrlrange[:, 1].copy()
        self._base_pos = np.array([0.0, 0.0, 0.16], dtype=np.float64)

        obs_dim = self._joint_count + self._joint_count + 3 + 1 + 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self._joint_count,), dtype=np.float32)
        if self.goal_conditioned:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                    "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                }
            )
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._step_count = 0
        self._prev_distance = 0.0
        self._prev_action = np.zeros(self._joint_count, dtype=np.float64)
        self._initial_distance = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._sample_valid_scene()
        self._step_count = 0
        self._prev_distance = self._target_distance()
        self._prev_action = np.zeros(self._joint_count, dtype=np.float64)
        self._initial_distance = self._prev_distance
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        target_ctrl = self._action_to_ctrl(action)
        self.data.ctrl[:] = target_ctrl

        substeps = max(1, int(round(self.config.control_dt / self.model.opt.timestep)))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        distance = info["distance_to_target"]
        previous_distance = self._prev_distance
        reward = self.compute_reward(info["ee_position"], info["target_position"], info)
        self._prev_distance = distance

        if not self.goal_conditioned:
            reward += self.config.distance_reward_scale * (previous_distance - distance)
            reward -= self.config.action_penalty * float(np.sum(np.square(action)))
            if self.config.action_delta_penalty > 0.0:
                reward -= self.config.action_delta_penalty * float(np.sum(np.square(action - self._prev_action)))
        self._prev_action = action.copy()

        collision = info["collision"]
        success = distance <= self.config.goal_tolerance
        truncated = self._step_count >= self.config.episode_steps
        terminated = collision or success

        if not self.goal_conditioned:
            if collision:
                reward -= self.config.collision_penalty
            if success:
                reward += self.config.success_reward

        info["success"] = success
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
        desired_goal = np.asarray(desired_goal, dtype=np.float64)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward = -distance - self.config.time_penalty
        if isinstance(info, dict) and "min_obstacle_clearance" in info:
            reward = reward - self._clearance_penalty(float(info["min_obstacle_clearance"]))
            if bool(info.get("collision", False)):
                reward = reward - self.config.collision_penalty
        return reward.astype(np.float32) if isinstance(reward, np.ndarray) else float(reward)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer as mj_viewer

                self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, width=640, height=480)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        center = 0.5 * (self._ctrl_lower_bounds + self._ctrl_upper_bounds)
        half_range = 0.5 * (self._ctrl_upper_bounds - self._ctrl_lower_bounds)
        target = center + action * half_range * self.config.action_scale
        return np.clip(target, self._ctrl_lower_bounds, self._ctrl_upper_bounds)

    def _get_obs(self):
        qpos = self.data.qpos[: self._joint_count].copy()
        qvel = self.data.qvel[: self._joint_count].copy()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target = self.data.mocap_pos[0].copy()
        rel_target = target - ee_pos
        distance = np.array([np.linalg.norm(rel_target)], dtype=np.float64)
        sensors = self._local_obstacle_sensors()
        observation = np.concatenate([qpos, qvel, rel_target, distance, sensors]).astype(np.float32)
        if self.goal_conditioned:
            return {
                "observation": observation,
                "achieved_goal": ee_pos.astype(np.float32),
                "desired_goal": target.astype(np.float32),
            }
        return observation

    def _get_info(self) -> dict:
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target = self.data.mocap_pos[0].copy()
        return {
            "ee_position": ee_pos,
            "target_position": target,
            "initial_distance": self._initial_distance,
            "distance_to_target": float(np.linalg.norm(target - ee_pos)),
            "collision": self._has_collision(),
            "sensor_readings": self._local_obstacle_sensors(),
            "min_obstacle_clearance": self._min_obstacle_clearance(),
        }

    def _target_distance(self) -> float:
        return float(np.linalg.norm(self.data.mocap_pos[0] - self.data.site_xpos[self.ee_site_id]))

    def _place_target(self) -> None:
        candidate = self._sample_workspace_target()
        if candidate is not None:
            self.data.mocap_pos[0] = candidate
        else:
            self.data.mocap_pos[0] = np.array([0.48, 0.0, 0.42], dtype=np.float64)
        self.data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _place_obstacles(self) -> None:
        placed_positions: list[np.ndarray] = []
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self.data.mocap_pos[0].copy()

        for i in range(len(self.obstacle_body_ids)):
            if i < self.config.obstacle_count:
                pos = self._sample_obstacle_position(ee_pos, target_pos, placed_positions)
                placed_positions.append(pos)
            else:
                pos = np.array([2.0 + i, 2.0, 2.0], dtype=np.float64)
            self.data.mocap_pos[1 + i] = pos
            self.data.mocap_quat[1 + i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _sample_valid_scene(self) -> None:
        fallback_qpos = self._home_qpos.copy()
        for _ in range(self.config.max_reset_tries):
            self.data.qpos[:] = self._sample_joint_configuration()
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = self.data.qpos[: self._joint_count]
            mujoco.mj_forward(self.model, self.data)

            self._place_target()
            self._place_obstacles()
            mujoco.mj_forward(self.model, self.data)

            if not self._has_collision() and self._target_distance() >= self.config.min_target_ee_distance:
                return

        self.data.qpos[:] = fallback_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.data.qpos[: self._joint_count]
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = np.array([0.56, 0.0, 0.42], dtype=np.float64)
        self.data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for i in range(self.config.obstacle_count):
            self.data.mocap_pos[1 + i] = np.array([0.38 + 0.1 * i, (-1) ** i * 0.16, 0.34], dtype=np.float64)
            self.data.mocap_quat[1 + i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for i in range(self.config.obstacle_count, len(self.obstacle_body_ids)):
            self.data.mocap_pos[1 + i] = np.array([2.0 + i, 2.0, 2.0], dtype=np.float64)
            self.data.mocap_quat[1 + i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _sample_joint_configuration(self) -> np.ndarray:
        low = np.maximum(self._joint_lower_bounds, self._home_qpos - self._reset_qpos_jitter)
        high = np.minimum(self._joint_upper_bounds, self._home_qpos + self._reset_qpos_jitter)
        return self.np_random.uniform(low=low, high=high)

    def _sample_workspace_target(self) -> np.ndarray | None:
        current_ee = self.data.site_xpos[self.ee_site_id].copy()
        for _ in range(self.config.reachable_target_samples):
            candidate = self._sample_spherical_target()
            if np.linalg.norm(candidate - self._base_pos) < self.config.min_target_base_distance:
                continue
            if np.linalg.norm(candidate - current_ee) < self.config.min_target_ee_distance:
                continue
            return candidate
        return None

    def _sample_spherical_target(self) -> np.ndarray:
        x = self.np_random.uniform(*self.config.target_x_bounds)
        y = self.np_random.uniform(*self.config.target_y_bounds)
        z = self.np_random.uniform(*self.config.target_z_bounds)
        return np.array([x, y, z], dtype=np.float64)

    def _sample_obstacle_position(
        self,
        ee_pos: np.ndarray,
        target_pos: np.ndarray,
        placed_positions: list[np.ndarray],
    ) -> np.ndarray:
        for _ in range(self.config.max_reset_tries):
            candidate = np.array(
                [
                    self.np_random.uniform(*self.config.obstacle_x_bounds),
                    self.np_random.uniform(*self.config.obstacle_y_bounds),
                    self.np_random.uniform(*self.config.obstacle_z_bounds),
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(candidate - ee_pos) < self.config.min_obstacle_ee_distance:
                continue
            if np.linalg.norm(candidate - target_pos) < self.config.min_obstacle_target_distance:
                continue
            if any(np.linalg.norm(candidate - other) < self.config.min_obstacle_spacing for other in placed_positions):
                continue
            return candidate

        candidate = np.array([0.44, 0.20 if not placed_positions else -0.20, 0.36], dtype=np.float64)
        if placed_positions:
            candidate[0] += 0.08 * len(placed_positions)
        return candidate

    def _local_obstacle_sensors(self) -> np.ndarray:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        distances = []
        active = min(self.config.obstacle_count, len(self.obstacle_body_ids))
        for i in range(active):
            obstacle_pos = self.data.mocap_pos[1 + i]
            distances.append(np.linalg.norm(obstacle_pos - ee_pos) - self.config.obstacle_radius)
        distances.sort()
        readings = [np.clip(distance / self.config.sensor_range, 0.0, 1.0) for distance in distances[:3]]
        while len(readings) < 3:
            readings.append(1.0)
        return np.asarray(readings, dtype=np.float64)

    def _min_obstacle_clearance(self) -> float:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        active = min(self.config.obstacle_count, len(self.obstacle_body_ids))
        if active == 0:
            return self.config.sensor_range
        clearances = []
        for i in range(active):
            obstacle_pos = self.data.mocap_pos[1 + i]
            clearances.append(np.linalg.norm(obstacle_pos - ee_pos) - self.config.obstacle_radius)
        return float(min(clearances))

    def _clearance_penalty(self, min_clearance: float) -> float:
        gap = self.config.clearance_threshold - min_clearance
        if gap <= 0.0:
            return 0.0
        return self.config.clearance_penalty_scale * gap

    def _has_collision(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name
            name1 = geom1 or ""
            name2 = geom2 or ""
            obstacle_hit = name1.startswith("obstacle_") or name2.startswith("obstacle_")
            if not obstacle_hit:
                continue
            arm_hit = self._is_arm_geom(name1) or self._is_arm_geom(name2)
            if not arm_hit:
                other_name = name2 if name1.startswith("obstacle_") else name1
                arm_hit = not other_name.startswith(("floor", "target", "obstacle_"))
            if arm_hit and obstacle_hit:
                return True
        return False

    @staticmethod
    def _is_arm_geom(geom_name: str) -> bool:
        return (
            geom_name.startswith("link")
            or geom_name.startswith("ee_")
            or "wrist" in geom_name
            or "shoulder" in geom_name
            or "upperarm" in geom_name
            or "forearm" in geom_name
            or "base_" in geom_name
        )
