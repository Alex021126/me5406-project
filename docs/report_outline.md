# Individual Report Outline

Use this as a clean starting structure for the 10-page individual report. Rewrite everything in your own words.

## 1. Introduction

- Motivate obstacle-aware manipulation in cluttered workspaces.
- Explain why reactive control matters for robotics.
- Briefly summarize conventional alternatives such as IK and sampling-based planning.

## 2. Problem Formulation

- Describe the 3-DOF arm, target-reaching task, and obstacle setup.
- Define the observation space, action space, and termination conditions.
- Explain why the local sensing assumption is robotically meaningful.

## 3. Reinforcement Learning Method

- Present the SAC algorithm at a high level.
- Describe the policy and critic network structure.
- Explain the reward terms and why each one was needed.
- Document key hyperparameters and training duration.

## 4. Experimental Setup

- Describe hardware/software used.
- State MuJoCo, Python, PyTorch, and Stable-Baselines3 versions.
- Define test scenarios for 1, 3, and 5 obstacles.
- Explain evaluation metrics: success rate, collision rate, return, and latency.

## 5. Results

- Present quantitative results for SAC.
- Compare with the IK baseline.
- Include representative screenshots or trajectory plots.
- Discuss when SAC works well and where it fails.

## 6. Discussion

- Advantages of RL in this task.
- Current limitations of the simplified sensing and arm model.
- Why the baseline is still useful for analysis.

## 7. Challenges and Lessons Learned

- Reward shaping issues.
- Instability during early training.
- Simulation design tradeoffs.
- Team workflow and experiment management.

## 8. Future Work

- Better obstacle sensing such as ray-casting or depth maps.
- Gripper extension for actual grasping.
- Stronger planners such as RRT* or OMPL-based baselines.
- Domain randomization and sim-to-real considerations.

## 9. Disclosure of External Code

- List any repo, tutorial, or code snippet that influenced the implementation.
- State clearly what was reused and what was implemented from scratch.
