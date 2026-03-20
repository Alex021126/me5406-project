from __future__ import annotations

from dataclasses import asdict

import numpy as np
from stable_baselines3 import SAC

from .envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


def evaluate_sac(model_path: str, episodes: int = 20, obstacle_count: int = 3) -> dict:
    env = ObstacleAvoidanceArmEnv(config=EnvConfig(obstacle_count=obstacle_count))
    model = SAC.load(model_path)

    successes = 0
    collisions = 0
    returns = []
    steps = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        episode_steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            episode_steps += 1

        successes += int(info["success"])
        collisions += int(info["collision"])
        returns.append(episode_return)
        steps.append(episode_steps)

    env.close()
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_episode_steps": float(np.mean(steps)),
        "config": asdict(env.config),
    }
