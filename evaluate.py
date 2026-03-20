import json
import sys

from me5406_project.evaluation import evaluate_sac


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python evaluate.py <model_path>")
    metrics = evaluate_sac(sys.argv[1])
    print(json.dumps(metrics, indent=2))
