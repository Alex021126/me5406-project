from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from .envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


def make_env(obstacle_count: int = 3, render_mode: str | None = None):
    def _factory():
        env = ObstacleAvoidanceArmEnv(
            render_mode=render_mode,
            config=EnvConfig(obstacle_count=obstacle_count),
        )
        return Monitor(env)

    return _factory


def train_sac(
    total_timesteps: int = 500_000,
    obstacle_count: int = 3,
    model_dir: str = "artifacts/models",
    seed: int = 42,
    device: str | None = None,
) -> Path:
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = "artifacts/tb" if importlib.util.find_spec("tensorboard") is not None else None
    progress_bar = (
        importlib.util.find_spec("tqdm") is not None and importlib.util.find_spec("rich") is not None
    )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {resolved_device}")

    env = make_vec_env(make_env(obstacle_count=obstacle_count), n_envs=1)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=200_000,
        gamma=0.99,
        tau=0.005,
        learning_starts=5_000,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs={"net_arch": {"pi": [256, 256], "qf": [256, 256]}},
        seed=seed,
        device=resolved_device,
    )

    checkpoint_dir = Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointCallback(save_freq=10_000, save_path=str(checkpoint_dir), name_prefix="sac_arm")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint, progress_bar=progress_bar)

    model_path = out_dir / f"sac_arm_obs{obstacle_count}_seed{seed}.zip"
    model.save(str(model_path))
    env.close()
    return model_path
