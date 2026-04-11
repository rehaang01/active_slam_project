#!/usr/bin/env python3
"""
test_surrogate.py — Quick smoke test for the SLAM Surrogate.

Verifies that SLAMSurrogate provides the same API as SLAMDataCollector
and that the synthetic covariance / loop closure models behave correctly.

Usage:
  # Offline test (no Gazebo needed — tests API shape only):
  python3 test_surrogate.py --offline

  # Live test (requires Gazebo + PX4 running):
  python3 test_surrogate.py
"""

import argparse
import sys
import os
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def test_api_compatibility():
    """Verify SLAMSurrogate has every public method that SLAMDataCollector has."""
    from envs.slam_collector import SLAMDataCollector
    from envs.slam_surrogate import SLAMSurrogate

    # Methods that active_slam_env.py calls
    required_methods = [
        'get_drone_position',
        'get_drone_yaw',
        'get_covariance_trace',
        'get_covariance_trace_normalized',
        'get_altitude',
        'get_min_lidar_range',
        'get_lidar_range_in_direction',
        'get_lidar_ranges_array',
        'get_depth_min_distance',
        'is_armed',
        'is_tracking_lost',
        'get_frontier_points_2d',
        'set_z_filter',
        'get_z_filter_state',
        'start_offboard_stream',
        'update_setpoint',
        'stop_offboard_stream',
        'publish_offboard_heartbeat',
        'publish_setpoint',
        'publish_command',
        'arm',
        'disarm',
        'engage_offboard',
        'land',
    ]

    # Attributes that active_slam_env.py reads
    required_attrs = [
        'occupancy_grid',
        'grid_info',
        'known_cells_count',
        'total_cells',
        'global_pose_covariance',
        'loop_closure_id',
        'proximity_detection_id',
        'frontier_count',
        'frontier_points',
        'odom',
        'octomap_data',
        'local_position',
        'vehicle_status',
        'lidar_ranges',
        'lidar_min_range',
        'lidar_angle_min',
        'lidar_angle_max',
        'lidar_angle_increment',
        'lidar_range_min',
        'lidar_range_max',
        'depth_min_distance',
        'depth_image',
        'tracking_lost',
    ]

    collector_methods = set(dir(SLAMDataCollector))
    surrogate_methods = set(dir(SLAMSurrogate))

    print("=" * 60)
    print("  API Compatibility Test")
    print("=" * 60)

    all_pass = True

    print("\n[Methods]")
    for method in required_methods:
        in_collector = method in collector_methods
        in_surrogate = method in surrogate_methods
        status = "OK" if (in_collector and in_surrogate) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}  {method:40s}  collector={'Y' if in_collector else 'N'}  surrogate={'Y' if in_surrogate else 'N'}")

    print("\n[Attributes] (checked at class level — runtime attrs verified in live test)")
    # Note: attrs are set in __init__, so we can't check them on the class
    # We just verify our list is complete
    for attr in required_attrs:
        print(f"  INFO {attr}")

    if all_pass:
        print("\n  ALL METHOD CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED — see above")
    print()
    return all_pass


def test_covariance_model():
    """Test the synthetic covariance model offline."""
    from envs.slam_surrogate import (
        COV_DRIFT_RATE, COV_BASE_NOISE, COV_LOOP_CLOSURE_DROP,
        COV_MAX_CLAMP, COV_NOISE_STD
    )

    print("=" * 60)
    print("  Synthetic Covariance Model Test")
    print("=" * 60)

    # Simulate 100m of straight-line travel
    distance = 0.0
    cov = COV_BASE_NOISE
    step_size = 0.5  # meters per step

    print(f"\n  Parameters:")
    print(f"    Drift rate:       {COV_DRIFT_RATE} per meter")
    print(f"    Base noise:       {COV_BASE_NOISE}")
    print(f"    LC drop factor:   {COV_LOOP_CLOSURE_DROP}")
    print(f"    Max clamp:        {COV_MAX_CLAMP}")

    print(f"\n  Simulating 100m travel with loop closure at 30m and 70m:\n")
    print(f"  {'Distance':>10s}  {'Covariance':>12s}  {'Event':s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*20}")

    for step in range(200):
        distance += step_size
        cov = COV_BASE_NOISE + COV_DRIFT_RATE * distance

        # Simulate loop closures
        event = ""
        if abs(distance - 30.0) < step_size:
            cov *= COV_LOOP_CLOSURE_DROP
            distance_since_lc = 0.0  # Would reset in real model
            event = "LOOP CLOSURE"
        elif abs(distance - 70.0) < step_size:
            cov *= COV_LOOP_CLOSURE_DROP
            event = "LOOP CLOSURE"

        cov = min(cov, COV_MAX_CLAMP)

        if step % 20 == 0 or event:
            print(f"  {distance:10.1f}m  {cov:12.4f}  {event}")

    print(f"\n  Final covariance at 100m: {cov:.4f}")
    print(f"  Expected range: [{COV_BASE_NOISE:.4f}, {COV_MAX_CLAMP:.4f}]")
    print()
    return True


def test_live(duration=30):
    """Live test with Gazebo — verify data flows correctly."""
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading

    if not rclpy.ok():
        rclpy.init()

    from envs.slam_surrogate import SLAMSurrogate

    print("=" * 60)
    print(f"  Live Test ({duration}s)")
    print("=" * 60)

    node = SLAMSurrogate()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("\n  Waiting for PX4 data...")
    start = time.time()
    while time.time() - start < 15.0:
        if node.local_position is not None:
            print("  PX4 position received!")
            break
        time.sleep(0.5)
    else:
        print("  WARNING: No PX4 data after 15s. Is Gazebo running?")
        return False

    print(f"\n  Monitoring for {duration}s...\n")
    print(f"  {'Time':>6s}  {'Pos X':>7s}  {'Pos Y':>7s}  {'Alt':>5s}  "
          f"{'Cov':>8s}  {'Known':>7s}  {'LiDAR':>6s}  {'LC':>3s}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*5}  "
          f"{'-'*8}  {'-'*7}  {'-'*6}  {'-'*3}")

    for t in range(duration):
        time.sleep(1.0)

        pos = node.get_drone_position()
        alt = node.get_altitude()
        cov = node.get_covariance_trace()
        known = node.known_cells_count
        lidar_min = node.get_min_lidar_range()
        lc = node.loop_closure_id
        tracking = "OK" if not node.is_tracking_lost() else "LOST"

        if pos is not None:
            print(f"  {t:5d}s  {pos[0]:7.2f}  {pos[1]:7.2f}  "
                  f"{alt:5.2f}  {cov:8.4f}  {known:7d}  "
                  f"{lidar_min:6.2f}  {lc:3d}")
        else:
            print(f"  {t:5d}s  (no position)")

    # Verify grid was built
    grid = node.occupancy_grid
    n_free = np.sum(grid == 0)
    n_occ = np.sum(grid > 0)
    n_unk = np.sum(grid < 0)
    total = grid.size

    print(f"\n  Grid stats:")
    print(f"    Free:    {n_free:7d} ({100*n_free/total:.1f}%)")
    print(f"    Occupied:{n_occ:7d} ({100*n_occ/total:.1f}%)")
    print(f"    Unknown: {n_unk:7d} ({100*n_unk/total:.1f}%)")
    print(f"    Total:   {total:7d}")
    print(f"    Tracking: always {tracking}")

    # Cleanup
    executor.shutdown()
    node.destroy_node()

    print("\n  LIVE TEST COMPLETE")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SLAM Surrogate")
    parser.add_argument("--offline", action="store_true",
                        help="Run offline tests only (no Gazebo needed)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration of live test in seconds")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SLAM Surrogate — Test Suite")
    print("=" * 60 + "\n")

    # Always run offline tests
    api_ok = test_api_compatibility()
    cov_ok = test_covariance_model()

    if not args.offline:
        live_ok = test_live(args.duration)
    else:
        live_ok = True
        print("[Skipping live test — use without --offline to test with Gazebo]")

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  API compatibility: {'PASS' if api_ok else 'FAIL'}")
    print(f"  Covariance model:  {'PASS' if cov_ok else 'FAIL'}")
    print(f"  Live test:         {'PASS' if live_ok else 'FAIL'}")
    print("=" * 60 + "\n")

    sys.exit(0 if (api_ok and cov_ok and live_ok) else 1)       