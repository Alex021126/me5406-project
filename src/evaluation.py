from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import zipfile

import numpy as np
import torch
from stable_baselines3 import SAC

from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


def model_uses_goal_conditioning(model_path: str) -> bool:
    with zipfile.ZipFile(model_path) as archive:
        data = archive.read("data").decode("utf-8")
    return "MultiInputPolicy" in data or "HerReplayBuffer" in data


def evaluate_sac(
    model_path: str,
    episodes: int = 20,
    obstacle_count: int = 3,
    device: str | None = None,
) -> dict:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {resolved_device}")
    goal_conditioned = model_uses_goal_conditioning(model_path)
    env = ObstacleAvoidanceArmEnv(
        config=EnvConfig(obstacle_count=obstacle_count),
        goal_conditioned=goal_conditioned,
    )
    model = SAC.load(model_path, env=env, device=resolved_device)

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
        "model_path": model_path,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_episode_steps": float(np.mean(steps)),
        "return_std": float(np.std(returns)),
        "step_std": float(np.std(steps)),
        "goal_conditioned": goal_conditioned,
        "config": asdict(env.config),
    }


def save_metrics(metrics: dict, output_path: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output
