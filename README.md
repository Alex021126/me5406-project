# ME5406 Project 2

This repository implements a course project scaffold for **Obstacle Avoidance Grasping for Robot Arm Based on Deep Reinforcement Learning**. It is structured to match the typical final submission requirements: training code, evaluation code, reproducible environment files, baseline hooks, and documentation.

## What is included

- A Universal Robots UR5e MuJoCo model imported from Google DeepMind's MuJoCo Menagerie.
- A Gymnasium environment with:
  - joint-angle and joint-velocity observations,
  - relative target position and Euclidean distance,
  - local obstacle sensing near the end effector,
  - dense reward shaping, collision penalty, and success reward.
- Goal-conditioned SAC + HER training using Stable-Baselines3.
- Evaluation helpers and simple baseline hooks for IK and future RRT* comparison.
- Experiment automation for obstacle-density sweeps and JSON/CSV metric export.
- Report and video support templates under `docs/`.

## Project layout

- `src/assets/ur5e_obstacle_scene.xml`: UR5e task scene with target and obstacle mocap bodies.
- `src/assets/universal_robots_ur5e/`: UR5e model and mesh assets from MuJoCo Menagerie.
- `src/envs/obstacle_avoidance_env.py`: custom environment.
- `src/training.py`: SAC training entrypoint.
- `src/evaluation.py`: trained-policy evaluation.
- `src/baselines.py`: baseline utilities.
- `train.py`: quick training script.
- `resume_train.py`: continue training from an existing SAC checkpoint.
- `evaluate.py`: evaluation CLI.
- `visualize.py`: render a trained policy and optionally save a GIF.
- `scripts/visualize_workspace.py`: sample and visualize the arm workspace.
- `scripts/run_experiment_suite.py`: train/evaluate SAC and IK across obstacle densities.
- `requirements.txt` and `environment.yml`: reproducible environments.
- `docs/report_outline.md`: individual report starter.
- `docs/video_checklist.md`: group video checklist.
- `SUBMISSION_CHECKLIST.md`: final packaging checklist.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Alternative Conda setup:

```bash
conda env create -f environment.yml
conda activate me5406-drl-arm
pip install -e .
```

MuJoCo 3.x is installed through `pip`. If your machine needs OpenGL support for human rendering, use offscreen evaluation first. On Ubuntu, common system packages are:

```bash
sudo apt update
sudo apt install -y libgl1 libglew2.2 libglfw3 libosmesa6
```

## Train

```bash
python train.py --timesteps 500000 --obstacles 1 --seed 42
```

This saves a trained model under `artifacts/models/` and intermediate checkpoints under `artifacts/checkpoints/`.
After changing environment randomization or reward settings, retrain the model instead of reusing older `.zip` files from previous runs.
Recommended curriculum: train and verify `1` obstacle first, then repeat for `3`, and only then scale to `5`.

## Resume training

```bash
python resume_train.py artifacts/models/sac_her_ur5e_obs1_seed42.zip --timesteps 100000 --obstacles 1
```

## Evaluate

```bash
python evaluate.py artifacts/models/sac_her_ur5e_obs1_seed42.zip --episodes 20 --obstacles 1 --output artifacts/results/eval_ur5e_obs1.json
```

## Run the experiment suite

This script trains SAC and evaluates both SAC and the IK baseline for obstacle counts `1 3 5`, then exports JSON and CSV summaries:

```bash
python scripts/run_experiment_suite.py --timesteps 100000 --episodes 10 --obstacles 1 3 5
```

Main outputs:

- `artifacts/models/`: trained SAC policies.
- `artifacts/results/*.json`: per-controller metrics.
- `artifacts/results/summary.csv`: compact comparison table.

## Visualize a rollout

Render the latest trained model in a local window:

```bash
mjpython visualize.py --human --obstacles 3
```

Save a GIF for your report or presentation:

```bash
python visualize.py --obstacles 1 --episodes 1 --save-gif artifacts/results/rollout_obs1.gif
```

On macOS, `--human` requires `mjpython` because of MuJoCo's viewer backend. GIF export works with normal `python`.

## Visualize the workspace

Generate a 3D scatter plot of sampled end-effector positions:

```bash
python scripts/visualize_workspace.py --samples 5000 --output artifacts/results/workspace_ur5e.png
```

## Design assumptions

The proposal leaves several implementation details unspecified. This scaffold makes the following explicit choices:

- Obstacles are represented as spheres with randomized positions in front of the arm.
- Local obstacle sensing is modeled as three normalized distance readings from the nearest active obstacles to the end effector.
- The action space is normalized continuous UR5e joint-position increments in `[-1, 1]`, mapped to small per-step position targets.
- The target is a point-reaching objective. Gripper closure is not modeled yet.
- Collision detection focuses on arm-vs-obstacle contacts.
- Reset logic now samples targets approximately uniformly in a Cartesian workspace box, then filters them by reachability and collision validity, so training/evaluation/visualization share the same reachability-aware scene generator.
- Reward shaping is intentionally simple: progress toward the goal, a small control penalty, a light near-obstacle penalty, a large collision penalty, and a large success bonus.
- The current training setup also enforces a larger initial target distance to avoid trivial 1-2 step successes.

## Suggested next extensions

1. Replace the simplified obstacle sensor vector with ray-casting or depth-map observations.
2. Add a proper RRT* baseline using OMPL or a custom planner.
3. Extend the task from reaching to grasping by adding a gripper and object attachment logic.
4. Sweep obstacle densities `1, 3, 5` and export tables for success rate, latency, collision rate, and smoothness.

## Submission notes

- Include at least one trained model in `artifacts/models/` before submission.
- Put your final evaluation outputs in `artifacts/results/`.
- Use `docs/report_outline.md` only as a structure guide; do not copy proposal text directly into the final report.
- A practical sequence is `obs=1` for debugging, then `obs=3` for the main reported result, and `obs=5` as the harder stress test.
