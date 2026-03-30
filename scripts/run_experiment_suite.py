from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.baselines import evaluate_ik_baseline, evaluate_rrt_star_baseline
from src.evaluation import evaluate_sac, save_metrics
from src.training import train_sac


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate SAC across obstacle densities.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obstacles", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rrt-iterations", type=int, default=3000,
                        help="RRT* tree expansion budget per episode")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.csv"

    rows: list[dict] = []
    for obstacle_count in args.obstacles:
        model_path = train_sac(
            total_timesteps=args.timesteps,
            obstacle_count=obstacle_count,
            model_dir=args.model_dir,
            seed=args.seed,
            device=args.device,
        )
        sac_metrics = evaluate_sac(
            str(model_path),
            episodes=args.episodes,
            obstacle_count=obstacle_count,
            device=args.device,
        )
        save_metrics(sac_metrics, str(results_dir / f"sac_obs{obstacle_count}.json"))

        ik_metrics = evaluate_ik_baseline(episodes=args.episodes, obstacle_count=obstacle_count)
        save_metrics(ik_metrics, str(results_dir / f"ik_obs{obstacle_count}.json"))

        rrt_metrics = evaluate_rrt_star_baseline(
            episodes=args.episodes,
            obstacle_count=obstacle_count,
            max_iterations=args.rrt_iterations,
        )
        save_metrics(rrt_metrics, str(results_dir / f"rrt_obs{obstacle_count}.json"))

        rows.append(
            {
                "controller": "SAC",
                "obstacle_count": obstacle_count,
                "success_rate": sac_metrics["success_rate"],
                "collision_rate": sac_metrics["collision_rate"],
                "mean_return": sac_metrics["mean_return"],
                "mean_episode_steps": sac_metrics["mean_episode_steps"],
                "mean_planning_latency_sec": 0.0,
            }
        )
        rows.append(
            {
                "controller": "IK",
                "obstacle_count": obstacle_count,
                "success_rate": ik_metrics["success_rate"],
                "collision_rate": ik_metrics["collision_rate"],
                "mean_return": ik_metrics["mean_return"],
                "mean_episode_steps": ik_metrics["mean_episode_steps"],
                "mean_planning_latency_sec": f'{ik_metrics["mean_planning_latency_sec"]:.4f}',
            }
        )
        rows.append(
            {
                "controller": "RRT*",
                "obstacle_count": obstacle_count,
                "success_rate": rrt_metrics["success_rate"],
                "collision_rate": rrt_metrics["collision_rate"],
                "mean_return": rrt_metrics["mean_return"],
                "mean_episode_steps": rrt_metrics["mean_episode_steps"],
                "mean_planning_latency_sec": f'{rrt_metrics["mean_planning_latency_sec"]:.4f}',
            }
        )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "controller",
                "obstacle_count",
                "success_rate",
                "collision_rate",
                "mean_return",
                "mean_episode_steps",
                "mean_planning_latency_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
