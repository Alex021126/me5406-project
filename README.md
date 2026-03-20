# ME5406 Project 2

This repository scaffolds the proposal titled **Obstacle Avoidance Grasping for Robot Arm Based on Deep Reinforcement Learning** into a runnable MuJoCo + Gymnasium project.

## What is included

- A 3-DOF MuJoCo robot arm model.
- A Gymnasium environment with:
  - joint-angle and joint-velocity observations,
  - relative target position and Euclidean distance,
  - local obstacle sensing near the end effector,
  - dense reward shaping, collision penalty, and success reward.
- SAC training using Stable-Baselines3.
- Evaluation helpers and simple baseline hooks for IK and future RRT* comparison.

## Project layout

- `src/me5406_project/assets/arm_3dof.xml`: MuJoCo model.
- `src/me5406_project/envs/obstacle_avoidance_env.py`: custom environment.
- `src/me5406_project/training.py`: SAC training entrypoint.
- `src/me5406_project/evaluation.py`: trained-policy evaluation.
- `src/me5406_project/baselines.py`: baseline utilities.
- `train.py`: quick training script.
- `evaluate.py`: evaluation CLI.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

MuJoCo 3.x is installed through `pip`. If your machine needs OpenGL support for human rendering, use offscreen evaluation first.

## Train

```bash
python train.py
```

By default, this trains SAC with 3 active obstacles and saves checkpoints under `artifacts/models/`.

## Evaluate

```bash
python evaluate.py artifacts/models/sac_arm_obs3.zip
```

## Design assumptions

The proposal leaves several implementation details unspecified. This scaffold makes the following explicit choices:

- Obstacles are represented as spheres with randomized positions in front of the arm.
- Local obstacle sensing is modeled as three normalized distance readings from the end effector to the nearest active obstacles.
- The action space is continuous joint motor commands in `[-1, 1]`.
- The target is a point-reaching objective. Gripper closure is not modeled yet.
- Collision detection focuses on arm-vs-obstacle contacts.

## Suggested next extensions

1. Replace the simplified obstacle sensor vector with ray-casting or depth-map observations.
2. Add a proper RRT* baseline using OMPL or a custom planner.
3. Extend the task from reaching to grasping by adding a gripper and object attachment logic.
4. Sweep obstacle densities `1, 3, 5` and export tables for success rate, latency, collision rate, and smoothness.
