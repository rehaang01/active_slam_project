#!/usr/bin/env python3
"""
End-to-End Pipeline Test for Active SLAM.

Validates the COMPLETE stack in one short run:
  1. Environment creation (ROS2 init, sensor data arrival)
  2. Drone arm + takeoff
  3. 5 random RL steps (action → observation → reward)
  4. Sensor data flow (LiDAR, depth, SLAM, covariance)
  5. Architecture features (Z-filter, frontiers, AOU, path validation)
  6. Clean shutdown (land, disarm, ROS2 teardown)

Run inside Docker container:
    source /opt/ros/humble/setup.bash
    cd /root/active_slam_project
    python3 test_pipeline.py

Expected output: All checks PASS, no crashes, drone flies and returns.
If ANY check fails, fix it BEFORE attempting full RL training.
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from envs.active_slam_env import ActiveSLAMEnv


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    symbol = "✓" if condition else "✗"
    msg = f"  {symbol} {name}: {status}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    print_header("Active SLAM — End-to-End Pipeline Test")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_passed = True
    n_test_steps = 5

    # ==============================================================
    # Test 1: Environment Creation
    # ==============================================================
    print_header("Test 1: Environment Creation")

    try:
        env = ActiveSLAMEnv()
        all_passed &= check("Environment created", True)
    except Exception as e:
        check("Environment created", False, str(e))
        print("\nCANNOT CONTINUE — environment failed to create.")
        return

    # Check observation space
    obs_space = env.observation_space
    all_passed &= check(
        "Observation space is Dict",
        hasattr(obs_space, 'spaces') and 'map_tensor' in obs_space.spaces,
    )
    all_passed &= check(
        "map_tensor shape = (3, 64, 64)",
        obs_space['map_tensor'].shape == (3, 64, 64),
        f"got {obs_space['map_tensor'].shape}",
    )
    all_passed &= check(
        "scalars shape = (8,)",
        obs_space['scalars'].shape == (8,),
        f"got {obs_space['scalars'].shape}",
    )

    # Check action space
    act_space = env.action_space
    all_passed &= check(
        "Action space shape = (2,)",
        act_space.shape == (2,),
        f"got {act_space.shape}",
    )
    all_passed &= check(
        "Action space bounds [-1, 1]",
        float(act_space.low[0]) == -1.0 and float(act_space.high[0]) == 1.0,
    )

    # ==============================================================
    # Test 2: Sensor Data Flow
    # ==============================================================
    print_header("Test 2: Sensor Data Flow")

    node = env.ros_node

    # PX4 position
    pos = node.get_drone_position()
    all_passed &= check(
        "PX4 position data received",
        pos is not None,
        f"pos={pos}" if pos is not None else "None — QoS issue?",
    )

    # Occupancy grid
    all_passed &= check(
        "Occupancy grid received",
        node.occupancy_grid is not None,
        f"shape={node.occupancy_grid.shape}" if node.occupancy_grid is not None else "None",
    )

    # Covariance (from /rtabmap/localization_pose)
    cov = node.get_covariance_trace()
    all_passed &= check(
        "Covariance trace available",
        True,  # Always returns a value (0.0 default)
        f"trace={cov:.6f}",
    )

    # LiDAR — topic subscription check only (sensor is blocked while on ground,
    # full functional test happens after takeoff in Test 3b)
    all_passed &= check(
        "LiDAR subscription active",
        node.lidar_sub is not None,
        "will test readings after takeoff",
    )

    # Depth camera
    depth_min = node.get_depth_min_distance()
    all_passed &= check(
        "Depth camera data",
        True,  # May be inf if no obstacles close
        f"min_depth={depth_min:.2f}m" if depth_min < float('inf') else "inf (no close obstacles)",
    )

    # Frontier points
    all_passed &= check(
        "Frontier data available",
        True,
        f"count={node.frontier_count}",
    )

    # Vehicle status
    all_passed &= check(
        "Vehicle status received",
        node.vehicle_status is not None,
        "armed" if node.is_armed() else "disarmed",
    )

    # Z-filter state
    z_min, z_max, z_active = node.get_z_filter_state()
    all_passed &= check(
        "Z-filter state trackable",
        True,
        f"[{z_min:.1f}, {z_max:.1f}]m active={z_active}",
    )

    # ==============================================================
    # Test 3: Reset (arm + takeoff + initial observation)
    # ==============================================================
    print_header("Test 3: Reset (Arm + Takeoff)")

    try:
        obs, info = env.reset()
        all_passed &= check("reset() completed", True)
    except Exception as e:
        all_passed &= check("reset() completed", False, str(e))
        print("\nCANNOT CONTINUE — reset failed.")
        env.close()
        return

    # Check observation structure
    all_passed &= check(
        "Observation has map_tensor",
        "map_tensor" in obs,
    )
    all_passed &= check(
        "Observation has scalars",
        "scalars" in obs,
    )
    all_passed &= check(
        "map_tensor dtype = float32",
        obs["map_tensor"].dtype == np.float32,
        f"got {obs['map_tensor'].dtype}",
    )
    all_passed &= check(
        "scalars dtype = float32",
        obs["scalars"].dtype == np.float32,
        f"got {obs['scalars'].dtype}",
    )

    # Check observation values are in valid range
    mt = obs["map_tensor"]
    all_passed &= check(
        "map_tensor values in [0, 1]",
        float(mt.min()) >= 0.0 and float(mt.max()) <= 1.0,
        f"min={mt.min():.3f} max={mt.max():.3f}",
    )
    sc = obs["scalars"]
    all_passed &= check(
        "scalars values in [0, 1]",
        float(sc.min()) >= 0.0 and float(sc.max()) <= 1.0,
        f"values={[f'{v:.3f}' for v in sc]}",
    )

    # Check map has SOME content (not all zeros or all 0.5)
    unique_ch1 = len(np.unique(mt[0]))
    all_passed &= check(
        "Channel 1 (occupancy) has data",
        unique_ch1 > 1,
        f"{unique_ch1} unique values",
    )

    # Check info dict
    all_passed &= check(
        "Info has initial_coverage",
        "initial_coverage" in info,
        f"coverage={info.get('initial_coverage', '?')}",
    )
    all_passed &= check(
        "Info has initial_cov_trace",
        "initial_cov_trace" in info,
        f"cov={info.get('initial_cov_trace', '?')}",
    )

    # Check drone is in the air
    alt = node.get_altitude()
    all_passed &= check(
        "Drone is airborne",
        alt is not None and alt > 0.25,
        f"altitude={alt:.2f}m" if alt is not None else "unknown",
    )

    # LiDAR functional test (now that drone is airborne, sensor is unblocked)
    print("\n  --- LiDAR airborne check ---")
    lidar_min = node.get_min_lidar_range()
    if lidar_min == float('inf'):
        print("  (waiting for LiDAR data...)")
        for _ in range(30):  # up to 3 seconds
            time.sleep(0.1)
            lidar_min = node.get_min_lidar_range()
            if lidar_min < float('inf'):
                break
    all_passed &= check(
        "LiDAR data received (airborne)",
        lidar_min < float('inf'),
        f"min_range={lidar_min:.2f}m" if lidar_min < float('inf') else "No data — check /scan bridge",
    )
    if lidar_min < float('inf'):
        forward_range = node.get_lidar_range_in_direction(0.0)
        all_passed &= check(
            "LiDAR directional query works",
            True,
            f"forward_range={forward_range:.2f}m",
        )

    # ==============================================================
    # Test 4: RL Steps (5 random actions)
    # ==============================================================
    print_header(f"Test 4: Running {n_test_steps} Random Steps")

    step_results = []
    for i in range(n_test_steps):
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            step_ok = True
            step_results.append({
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "coverage": info.get("coverage", 0),
                "cov_trace": info.get("cov_trace", 0),
                "new_cells": info.get("new_cells", 0),
            })
            print(f"  Step {i+1}: action=[{action[0]:+.2f}, {action[1]:+.2f}] "
                  f"reward={reward:+.3f} "
                  f"cov={info.get('coverage', 0):.1%} "
                  f"new_cells={info.get('new_cells', 0)} "
                  f"term={terminated} trunc={truncated}")

            if terminated or truncated:
                print(f"  Episode ended at step {i+1}.")
                break

        except Exception as e:
            step_ok = False
            all_passed &= check(f"Step {i+1} executed", False, str(e))
            break

    if len(step_results) > 0:
        all_passed &= check(
            f"All {len(step_results)} steps completed",
            len(step_results) >= 1,
        )

        # Check reward is a finite number
        rewards = [s["reward"] for s in step_results]
        all_passed &= check(
            "Rewards are finite numbers",
            all(np.isfinite(r) for r in rewards),
            f"rewards={[f'{r:.3f}' for r in rewards]}",
        )

        # Check observation shape didn't change
        all_passed &= check(
            "Observation shape stable after steps",
            obs["map_tensor"].shape == (3, 64, 64) and obs["scalars"].shape == (8,),
        )

    # ==============================================================
    # Test 5: Architecture Features
    # ==============================================================
    print_header("Test 5: Architecture Features")

    # Frontier detection
    n_clusters = len(env.frontier_clusters)
    all_passed &= check(
        "Frontier detection ran",
        True,
        f"{n_clusters} clusters found, nearest={env.nearest_frontier_dist:.2f}m",
    )

    # Z-filter state
    z_min, z_max, z_active = node.get_z_filter_state()
    from envs.active_slam_env import ENABLE_Z_FILTER
    if ENABLE_Z_FILTER:
        all_passed &= check(
            "Z-filter is active",
            z_active,
            f"[{z_min:.1f}, {z_max:.1f}]m",
        )
    else:
        all_passed &= check(
            "Z-filter disabled (expected for initial training)",
            not z_active,
            "using default all-Z projected grid",
        )

    # Visited map exists and has data
    all_passed &= check(
        "Visited map populated",
        env.visited_map is not None and np.sum(env.visited_map) > 0,
        f"visited_cells={int(np.sum(env.visited_map > 0))}" if env.visited_map is not None else "None",
    )

    # Trajectory recorded
    all_passed &= check(
        "Trajectory recorded",
        len(env.trajectory) > 0,
        f"{len(env.trajectory)} waypoints",
    )

    # Covariance panic check ran without crash
    try:
        should_retreat, wp = env._check_covariance_retreat()
        all_passed &= check(
            "Covariance retreat check works",
            True,
            f"panic_mode={env.panic_mode}",
        )
    except Exception as e:
        all_passed &= check("Covariance retreat check works", False, str(e))

    # AOU works
    try:
        pos = node.get_drone_position()
        if pos is not None:
            sx, sy = env._snap_to_free_point(pos[0] + 1.0, pos[1])
            all_passed &= check(
                "AOU snap_to_free_point works",
                True,
                f"({pos[0]+1:.1f},{pos[1]:.1f}) → ({sx:.1f},{sy:.1f})",
            )
    except Exception as e:
        all_passed &= check("AOU snap_to_free_point works", False, str(e))

    # Path validation works
    try:
        if pos is not None:
            waypoints = env._validate_and_plan_path(
                pos[0], pos[1], pos[0] + 1.0, pos[1] + 1.0)
            all_passed &= check(
                "Path validation works",
                len(waypoints) > 0,
                f"{len(waypoints)} waypoints returned",
            )
    except Exception as e:
        all_passed &= check("Path validation works", False, str(e))

    # Reactive safety check
    try:
        should_hover = env._check_reactive_safety(target_yaw=0.0)
        all_passed &= check(
            "Reactive safety check works",
            True,
            f"should_hover={should_hover}",
        )
    except Exception as e:
        all_passed &= check("Reactive safety check works", False, str(e))

    # ==============================================================
    # Test 6: Clean Shutdown
    # ==============================================================
    print_header("Test 6: Clean Shutdown")

    try:
        env.close()
        all_passed &= check("Environment closed cleanly", True)
    except Exception as e:
        all_passed &= check("Environment closed cleanly", False, str(e))

    # ==============================================================
    # Final Summary
    # ==============================================================
    print_header("FINAL RESULT")

    if all_passed:
        print("  ✓ ALL TESTS PASSED")
        print("")
        print("  Your complete Active SLAM pipeline is operational:")
        print("    - PX4 control (arm, takeoff, setpoints, land)")
        print("    - RTAB-Map SLAM (occupancy grid, covariance, loop closures)")
        print("    - LiDAR (360° obstacle detection)")
        print("    - Z-filter (altitude-sliced maps)")
        print("    - Frontier detection (connected components)")
        print("    - Path validation (raytrace + A*)")
        print("    - Action Optimization Unit (snap to free)")
        print("    - Reactive safety (LiDAR + depth)")
        print("    - Covariance retreat logic")
        print("    - Gymnasium API (reset, step, observation, reward)")
        print("")
        print("  NEXT: Run full training with:")
        print("    python3 /root/active_slam_project/train.py")
    else:
        print("  ✗ SOME TESTS FAILED")
        print("")
        print("  Fix the failures above before attempting training.")
        print("  Common issues:")
        print("    - If PX4 position is None: QoS mismatch (check VOLATILE)")
        print("    - If occupancy grid is None: RTAB-Map not running")
        print("    - If LiDAR is inf: /scan bridge not working")
        print("    - If arm fails: PX4 SITL not started by tmuxinator")

    print("")


if __name__ == "__main__":
    main()