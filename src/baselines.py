from __future__ import annotations

import time
from statistics import mean, pstdev
from typing import List, Optional

import mujoco
import numpy as np

from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


# ── IK baseline ───────────────────────────────────────────────────────────────

def ik_velocity_baseline(env: ObstacleAvoidanceArmEnv, gain: float = 1.5) -> np.ndarray:
    rel_target = env.data.mocap_pos[0] - env.data.site_xpos[env.ee_site_id]
    jac_pos = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jac_pos, None, env.ee_site_id)
    dq = gain * np.linalg.pinv(jac_pos[:, :3]) @ rel_target
    return np.clip(dq, -1.0, 1.0)


def run_ik_episode(obstacle_count: int = 3, max_steps: int = 200) -> dict:
    env = ObstacleAvoidanceArmEnv(config=EnvConfig(obstacle_count=obstacle_count))
    obs, _ = env.reset()
    start = time.perf_counter()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0
    info = {}

    while not done and not truncated and steps < max_steps:
        action = ik_velocity_baseline(env)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    latency = time.perf_counter() - start
    env.close()
    return {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "return": total_reward,
        "steps": steps,
        "planning_latency_sec": latency,
    }


def evaluate_ik_baseline(episodes: int = 20, obstacle_count: int = 3) -> dict:
    runs = [run_ik_episode(obstacle_count=obstacle_count) for _ in range(episodes)]
    returns = [run["return"] for run in runs]
    steps = [run["steps"] for run in runs]
    latencies = [run["planning_latency_sec"] for run in runs]
    return {
        "episodes": episodes,
        "obstacle_count": obstacle_count,
        "success_rate": mean(int(run["success"]) for run in runs),
        "collision_rate": mean(int(run["collision"]) for run in runs),
        "mean_return": mean(returns),
        "return_std": pstdev(returns) if len(returns) > 1 else 0.0,
        "mean_episode_steps": mean(steps),
        "step_std": pstdev(steps) if len(steps) > 1 else 0.0,
        "mean_planning_latency_sec": mean(latencies),
    }


# ── RRT* baseline ─────────────────────────────────────────────────────────────


def _query_config(
    env: ObstacleAvoidanceArmEnv,
    q: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Set arm joints to q, run forward kinematics, return (ee_pos, collision)."""
    env.data.qpos[:3] = q
    env.data.qvel[:3] = 0.0
    mujoco.mj_forward(env.model, env.data)
    ee_pos = env.data.site_xpos[env.ee_site_id].copy()
    collision = env._has_collision()
    return ee_pos, collision


def _segment_free(
    env: ObstacleAvoidanceArmEnv,
    q_from: np.ndarray,
    q_to: np.ndarray,
    n_checks: int = 5,
) -> bool:
    """Return True if the straight-line segment in joint space is collision-free."""
    for i in range(1, n_checks + 1):
        t = i / n_checks
        _, coll = _query_config(env, q_from + t * (q_to - q_from))
        if coll:
            return False
    return True


def _ik_goal_config(
    env: ObstacleAvoidanceArmEnv,
    target_pos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    attempts: int = 20,
    ik_steps: int = 30,
) -> Optional[np.ndarray]:
    """Find a joint config that places the EE at target_pos via Jacobian IK.

    Uses random restarts so it is robust to local minima.  Returns the best
    config found (lowest EE error), or None if no attempt converges.
    """
    best_q: Optional[np.ndarray] = None
    best_dist = float("inf")
    for _ in range(attempts):
        q = rng.uniform(lower, upper)
        for _ in range(ik_steps):
            env.data.qpos[:3] = q
            env.data.qvel[:3] = 0.0
            mujoco.mj_forward(env.model, env.data)
            ee = env.data.site_xpos[env.ee_site_id].copy()
            rel = target_pos - ee
            dist = float(np.linalg.norm(rel))
            if dist < env.config.goal_tolerance:
                if dist < best_dist:
                    best_dist = dist
                    best_q = q.copy()
                break
            jac = np.zeros((3, env.model.nv))
            mujoco.mj_jacSite(env.model, env.data, jac, None, env.ee_site_id)
            dq = 0.3 * np.linalg.pinv(jac[:, :3]) @ rel
            q = np.clip(q + dq, lower, upper)
    return best_q


def rrt_star_plan(
    env: ObstacleAvoidanceArmEnv,
    max_iterations: int = 3000,
    step_size: float = 0.15,
    goal_bias: float = 0.10,
    rewire_radius: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> Optional[List[np.ndarray]]:
    """Plan a collision-free path in joint space from the current configuration
    to the target using RRT*.

    The tree is built in the 3-DOF joint space.  Goal-biased sampling uses an
    approximate IK solution so the tree grows toward the target efficiently.

    Args:
        env:            Environment (reset and ready).
        max_iterations: Maximum number of tree expansion attempts.
        step_size:      Max joint-space step length when steering (radians).
        goal_bias:      Probability of sampling near the IK goal config.
        rewire_radius:  Neighbourhood radius for the RRT* rewire step.
        rng:            NumPy random generator (created if None).

    Returns:
        List of joint-angle arrays [q_start, ..., q_goal], or None if the
        target was not reached within max_iterations.

    Note:
        This function freely modifies env.data.qpos/qvel.  The caller is
        responsible for saving and restoring env state.
    """
    if rng is None:
        rng = np.random.default_rng()

    q_start = env.data.qpos[:3].copy()
    target_pos = env.data.mocap_pos[0].copy()
    lower = env._joint_lower_bounds
    upper = env._joint_upper_bounds

    # Approximate goal config via IK for biased sampling
    q_goal_approx = _ik_goal_config(env, target_pos, lower, upper, rng)

    # Tree stored as parallel lists (more cache-friendly than a node class)
    nodes_q: List[np.ndarray] = [q_start.copy()]
    nodes_parent: List[int] = [-1]          # -1 = root sentinel
    nodes_cost: List[float] = [0.0]
    goal_node_idx: Optional[int] = None

    for _ in range(max_iterations):
        # ── Sample ───────────────────────────────────────────────────────────
        if q_goal_approx is not None and rng.random() < goal_bias:
            # Gaussian perturbation around the IK solution
            q_rand = np.clip(q_goal_approx + rng.normal(0.0, 0.05, 3), lower, upper)
        else:
            q_rand = rng.uniform(lower, upper)

        # ── Nearest node (vectorised) ─────────────────────────────────────────
        arr_q = np.array(nodes_q)                           # (N, 3)
        dists_all = np.linalg.norm(arr_q - q_rand, axis=1)  # (N,)
        nearest_idx = int(np.argmin(dists_all))
        q_near = nodes_q[nearest_idx]

        # ── Steer ────────────────────────────────────────────────────────────
        diff = q_rand - q_near
        dist_to_rand = float(np.linalg.norm(diff))
        if dist_to_rand < 1e-8:
            continue
        q_new = q_near + (min(step_size, dist_to_rand) / dist_to_rand) * diff
        q_new = np.clip(q_new, lower, upper)

        # ── Obstacle / segment check ──────────────────────────────────────────
        ee_new, coll_new = _query_config(env, q_new)
        if coll_new:
            continue
        if not _segment_free(env, q_near, q_new):
            continue

        # ── Neighbours within rewire radius ───────────────────────────────────
        dists_to_new = np.linalg.norm(arr_q - q_new, axis=1)   # reuse arr_q
        neighbor_idxs = np.where(dists_to_new < rewire_radius)[0][:20].tolist()

        # ── Choose best parent ────────────────────────────────────────────────
        best_parent = nearest_idx
        best_cost = nodes_cost[nearest_idx] + float(dists_to_new[nearest_idx])
        for ni in neighbor_idxs:
            c = nodes_cost[ni] + float(dists_to_new[ni])
            if c < best_cost and _segment_free(env, nodes_q[ni], q_new):
                best_cost = c
                best_parent = ni

        # ── Add node ──────────────────────────────────────────────────────────
        new_idx = len(nodes_q)
        nodes_q.append(q_new.copy())
        nodes_parent.append(best_parent)
        nodes_cost.append(best_cost)

        # ── Rewire neighbours ─────────────────────────────────────────────────
        for ni in neighbor_idxs:
            new_c = best_cost + float(dists_to_new[ni])
            if new_c < nodes_cost[ni] and _segment_free(env, q_new, nodes_q[ni]):
                nodes_parent[ni] = new_idx
                nodes_cost[ni] = new_c

        # ── Goal check ────────────────────────────────────────────────────────
        if float(np.linalg.norm(ee_new - target_pos)) <= env.config.goal_tolerance:
            if goal_node_idx is None or best_cost < nodes_cost[goal_node_idx]:
                goal_node_idx = new_idx

    if goal_node_idx is None:
        return None

    # Extract path by walking parent pointers back to root
    path: List[np.ndarray] = []
    idx = goal_node_idx
    while idx != -1:
        path.append(nodes_q[idx])
        idx = nodes_parent[idx]
    path.reverse()
    return path


def _smooth_path(
    env: ObstacleAvoidanceArmEnv,
    path: List[np.ndarray],
    n_passes: int = 2,
) -> List[np.ndarray]:
    """Greedy shortcutting: repeatedly attempt to skip intermediate waypoints.

    Each pass walks the path and tries to connect the current node directly to
    the furthest reachable node, reducing total waypoints.
    """
    for _ in range(n_passes):
        i = 0
        shortened = [path[i]]
        while i < len(path) - 1:
            # Try to jump as far ahead as possible
            j = len(path) - 1
            while j > i + 1:
                if _segment_free(env, path[i], path[j], n_checks=8):
                    break
                j -= 1
            shortened.append(path[j])
            i = j
        path = shortened
        if len(path) <= 2:
            break
    return path


def _follow_path(
    env: ObstacleAvoidanceArmEnv,
    path: List[np.ndarray],
    waypoint_tol: float = 0.08,
    max_steps_per_seg: int = 40,
    gain: float = 4.0,
) -> tuple[bool, bool, float, int]:
    """Execute a joint-space path via proportional velocity control.

    At each control step the action is ``clip(gain * (q_waypoint - q_current), -1, 1)``.
    The controller advances to the next waypoint once within ``waypoint_tol``.

    Returns:
        (success, collision, total_reward, total_steps)
    """
    total_reward = 0.0
    total_steps = 0
    for wp in path[1:]:    # path[0] is the start config already in the env
        for _ in range(max_steps_per_seg):
            q_curr = env.data.qpos[:3].copy()
            if float(np.linalg.norm(wp - q_curr)) < waypoint_tol:
                break
            action = np.clip(gain * (wp - q_curr), -1.0, 1.0).astype(np.float32)
            _, reward, done, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            if done or truncated:
                return (
                    bool(info.get("success", False)),
                    bool(info.get("collision", False)),
                    total_reward,
                    total_steps,
                )
    # Path exhausted without terminal signal – check final state
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    total_steps += 1
    total_reward += 0.0   # reward from zero action is tiny; ignore
    return (
        bool(info.get("success", False)),
        bool(info.get("collision", False)),
        total_reward,
        total_steps,
    )


def run_rrt_star_episode(
    obstacle_count: int = 3,
    max_iterations: int = 3000,
    seed: Optional[int] = None,
) -> dict:
    """Run one full RRT* episode: reset → plan → execute.

    Returns a metrics dict compatible with ``run_ik_episode``, plus
    ``path_found`` and ``path_length`` fields.
    """
    rng = np.random.default_rng(seed)
    env = ObstacleAvoidanceArmEnv(config=EnvConfig(obstacle_count=obstacle_count))
    env.reset(seed=int(rng.integers(0, 2**31)))

    # Save MuJoCo state before planning (planning mutates qpos/qvel freely)
    qpos_saved = env.data.qpos.copy()
    qvel_saved = env.data.qvel.copy()

    # ── Planning phase ────────────────────────────────────────────────────────
    plan_start = time.perf_counter()
    path = rrt_star_plan(env, max_iterations=max_iterations, rng=rng)
    if path is not None:
        path = _smooth_path(env, path)
    plan_time = time.perf_counter() - plan_start

    # Restore env to the post-reset state for execution
    env.data.qpos[:] = qpos_saved
    env.data.qvel[:] = qvel_saved
    mujoco.mj_forward(env.model, env.data)

    if path is None:
        env.close()
        return {
            "success": False,
            "collision": False,
            "return": 0.0,
            "steps": 0,
            "planning_latency_sec": plan_time,
            "path_found": False,
            "path_length": 0,
        }

    # ── Execution phase ───────────────────────────────────────────────────────
    success, collision, total_reward, steps = _follow_path(env, path)
    env.close()
    return {
        "success": success,
        "collision": collision,
        "return": total_reward,
        "steps": steps,
        "planning_latency_sec": plan_time,
        "path_found": True,
        "path_length": len(path),
    }


def evaluate_rrt_star_baseline(
    episodes: int = 20,
    obstacle_count: int = 3,
    max_iterations: int = 3000,
) -> dict:
    """Evaluate RRT* over N episodes and return aggregate metrics.

    The returned dict is compatible with ``evaluate_ik_baseline`` output and
    adds ``path_found_rate`` and ``latency_std``.
    """
    runs = [
        run_rrt_star_episode(
            obstacle_count=obstacle_count,
            max_iterations=max_iterations,
            seed=i,
        )
        for i in range(episodes)
    ]
    returns = [r["return"] for r in runs]
    steps = [r["steps"] for r in runs]
    latencies = [r["planning_latency_sec"] for r in runs]
    return {
        "episodes": episodes,
        "obstacle_count": obstacle_count,
        "success_rate": mean(int(r["success"]) for r in runs),
        "collision_rate": mean(int(r["collision"]) for r in runs),
        "path_found_rate": mean(int(r["path_found"]) for r in runs),
        "mean_return": mean(returns),
        "return_std": pstdev(returns) if len(returns) > 1 else 0.0,
        "mean_episode_steps": mean(steps),
        "step_std": pstdev(steps) if len(steps) > 1 else 0.0,
        "mean_planning_latency_sec": mean(latencies),
        "latency_std": pstdev(latencies) if len(latencies) > 1 else 0.0,
    }
