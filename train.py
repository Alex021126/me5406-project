import argparse

from me5406_project.training import train_sac


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on the 3-DOF obstacle avoidance arm task.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--obstacles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    path = train_sac(
        total_timesteps=args.timesteps,
        obstacle_count=args.obstacles,
        model_dir=args.model_dir,
        seed=args.seed,
        device=args.device,
    )
    print(f"Saved model to {path}")
