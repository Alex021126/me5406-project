# UR5e Obstacle 1 Archive

Archived on 2026-04-21 10:37 Asia/Singapore.

## Model

- `models/sac_her_ur5e_obs1_seed42.zip`
- Robot: Universal Robots UR5e from MuJoCo Menagerie
- Algorithm: SAC with HER replay
- Obstacles: 1
- Seed: 42

## Evaluation

20 episodes:

- Success rate: 0.95
- Collision rate: 0.05
- Mean return: -2.6671
- Mean episode steps: 4.35

100 episodes:

- Success rate: 0.96
- Collision rate: 0.07
- Mean return: -3.2647
- Mean episode steps: 4.32

## Included Files

- `results/eval_ur5e_obs1.json`
- `results/eval_ur5e_obs1_100ep.json`
- `results/rollout_ur5e_obs1.gif`
- `results/model_snapshot_ur5e.png`
- `config/ur5e_obstacle_scene.xml`
- `config/obstacle_avoidance_env.py`
- `config/training.py`
