from __future__ import annotations

import time
from statistics import mean, pstdev

import mujoco
import numpy as np

from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


def ik_velocity_baseline(env: ObstacleAvoidanceArmEnv, gain: float = 1.5) -> np.ndarray:
    rel_target = env.data.mocap_pos[0] - env.data.site_xpos[env.ee_site_id]
    jac_pos = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jac_pos, None, env.ee_site_id)
    dq = gain * np.linalg.pinv(jac_pos[:, : env.model.nu]) @ rel_target
    return np.clip(dq, -1.0, 1.0)


def run_ik_episode(obstacle_count: int = 3, max_steps: int = 200) -> dict:
    env = ObstacleAvoidanceArmEnv(config=EnvConfig(obstacle_count=obstacle_count))
    obs, _ = env.reset()
    start = time.perf_counter()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0
    info = {}

    while not done and not truncated and steps < max_steps:
        action = ik_velocity_baseline(env)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    latency = time.perf_counter() - start
    env.close()
    return {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "return": total_reward,
        "steps": steps,
        "planning_latency_sec": latency,
    }


def rrt_star_placeholder() -> dict:
    return {
        "implemented": False,
        "note": "RRT* baseline is left as a project extension. Use OMPL or a custom planner against the same workspace bounds.",
    }


def evaluate_ik_baseline(episodes: int = 20, obstacle_count: int = 3) -> dict:
    runs = [run_ik_episode(obstacle_count=obstacle_count) for _ in range(episodes)]
    returns = [run["return"] for run in runs]
    steps = [run["steps"] for run in runs]
    latencies = [run["planning_latency_sec"] for run in runs]
    return {
        "episodes": episodes,
        "obstacle_count": obstacle_count,
        "success_rate": mean(int(run["success"]) for run in runs),
        "collision_rate": mean(int(run["collision"]) for run in runs),
        "mean_return": mean(returns),
        "return_std": pstdev(returns) if len(returns) > 1 else 0.0,
        "mean_episode_steps": mean(steps),
        "step_std": pstdev(steps) if len(steps) > 1 else 0.0,
        "mean_planning_latency_sec": mean(latencies),
    }
