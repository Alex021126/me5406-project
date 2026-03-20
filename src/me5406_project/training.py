from __future__ import annotations

from pathlib import Path

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
    total_timesteps: int = 200_000,
    obstacle_count: int = 3,
    model_dir: str = "artifacts/models",
) -> Path:
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=str(out_dir / "tb"),
        policy_kwargs={"net_arch": {"pi": [256, 256], "qf": [256, 256]}},
    )

    checkpoint = CheckpointCallback(save_freq=10_000, save_path=str(out_dir), name_prefix="sac_arm")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint, progress_bar=True)

    model_path = out_dir / f"sac_arm_obs{obstacle_count}.zip"
    model.save(str(model_path))
    env.close()
    return model_path
