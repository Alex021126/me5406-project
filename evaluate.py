import argparse
import json
from pathlib import Path

from me5406_project.evaluation import evaluate_sac, save_metrics


def resolve_model_path(model_path: str | None) -> str:
    if model_path:
        return model_path

    candidates = sorted(Path("artifacts/models").glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            "No model_path was provided and no trained model was found in artifacts/models/. "
            "Train a model first or pass the .zip path explicitly."
        )
    return str(candidates[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC policy.")
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--obstacles", type=int, default=3)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    model_path = resolve_model_path(args.model_path)

    metrics = evaluate_sac(
        model_path,
        episodes=args.episodes,
        obstacle_count=args.obstacles,
        device=args.device,
    )
    if args.output:
        save_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2))
