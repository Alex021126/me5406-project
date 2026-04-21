from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


class GoalMonitor(Monitor):
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)


class EpisodeStatusCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue
            self.episode_count += 1
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            initial_distance = info.get("initial_distance")
            distance = info.get("distance_to_target")
            reward = info.get("episode", {}).get("r")
            length = info.get("episode", {}).get("l")
            initial_distance_text = (
                f"{initial_distance:.4f}" if isinstance(initial_distance, (int, float)) else "n/a"
            )
            distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "n/a"
            reward_text = f"{reward:.3f}" if isinstance(reward, (int, float)) else "n/a"
            length_text = str(length) if isinstance(length, int) else "n/a"
            print(
                f"Episode {self.episode_count}: "
                f"success={success} collision={collision} "
                f"length={length_text} return={reward_text} "
                f"initial_distance={initial_distance_text} final_distance={distance_text}"
            )
        return True


def make_env(obstacle_count: int = 3, render_mode: str | None = None, goal_conditioned: bool = False):
    def _factory():
        env = ObstacleAvoidanceArmEnv(
            render_mode=render_mode,
            config=EnvConfig(obstacle_count=obstacle_count),
            goal_conditioned=goal_conditioned,
        )
        if goal_conditioned:
            return GoalMonitor(env)
        return Monitor(env)

    return _factory


def train_sac(
    total_timesteps: int = 500_000,
    obstacle_count: int = 1,
    model_dir: str = "artifacts/models",
    seed: int = 42,
    device: str | None = None,
    use_her: bool = False,
) -> Path:
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = "artifacts/tb" if importlib.util.find_spec("tensorboard") is not None else None
    progress_bar = (
        importlib.util.find_spec("tqdm") is not None and importlib.util.find_spec("rich") is not None
    )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {resolved_device}")

    env = DummyVecEnv([make_env(obstacle_count=obstacle_count, goal_conditioned=use_her)])
    policy = "MultiInputPolicy" if use_her else "MlpPolicy"
    replay_buffer_kwargs = {"n_sampled_goal": 4, "goal_selection_strategy": "future"} if use_her else None
    model = SAC(
        policy,
        env,
        verbose=1,
        learning_rate=2e-4 if use_her else 3e-4,
        batch_size=256,
        buffer_size=500_000 if use_her else 200_000,
        gamma=0.99,
        tau=0.005,
        learning_starts=10_000 if use_her else 5_000,
        train_freq=1,
        gradient_steps=1,
        replay_buffer_class=HerReplayBuffer if use_her else None,
        replay_buffer_kwargs=replay_buffer_kwargs,
        tensorboard_log=tensorboard_log,
        policy_kwargs={"net_arch": {"pi": [256, 256], "qf": [256, 256]}},
        seed=seed,
        device=resolved_device,
    )

    checkpoint_dir = Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = "sac_her_ur5e" if use_her else "sac_ur5e"
    checkpoint = CheckpointCallback(save_freq=10_000, save_path=str(checkpoint_dir), name_prefix=checkpoint_prefix)
    callbacks = [checkpoint, EpisodeStatusCallback()]
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=progress_bar)

    prefix = "sac_her_ur5e" if use_her else "sac_ur5e"
    model_path = out_dir / f"{prefix}_obs{obstacle_count}_seed{seed}.zip"
    model.save(str(model_path))
    env.close()
    return model_path


def resume_sac(
    checkpoint_path: str,
    total_timesteps: int = 100_000,
    obstacle_count: int = 1,
    output_path: str | None = None,
    device: str | None = None,
) -> Path:
    tensorboard_log = "artifacts/tb" if importlib.util.find_spec("tensorboard") is not None else None
    progress_bar = (
        importlib.util.find_spec("tqdm") is not None and importlib.util.find_spec("rich") is not None
    )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {resolved_device}")

    env = DummyVecEnv([make_env(obstacle_count=obstacle_count)])
    model = SAC.load(checkpoint_path, env=env, device=resolved_device)
    model.tensorboard_log = tensorboard_log

    checkpoint_dir = Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointCallback(save_freq=10_000, save_path=str(checkpoint_dir), name_prefix="sac_arm_resume")
    callbacks = [checkpoint, EpisodeStatusCallback()]
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=progress_bar, reset_num_timesteps=False)

    if output_path is None:
        source = Path(checkpoint_path)
        output = source.with_name(f"{source.stem}_resumed.zip")
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    env.close()
    return output
