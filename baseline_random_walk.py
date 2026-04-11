#!/usr/bin/env python3
"""
Baseline: Random Walk Exploration

Strategy:
  At each step, sample a uniformly random (dx, dy) displacement within
  [-MAX_STEP_DIST, MAX_STEP_DIST]^2. If the sampled direction is blocked by
  LiDAR or depth camera, resample up to MAX_RESAMPLE_ATTEMPTS times before
  falling back to a stationary hover.

This is the weakest baseline — it makes no use of the occupancy map, frontier
information, or SLAM covariance. It serves as a lower bound for all metrics.

Metrics logged (identical schema to baseline_frontier.py):
  step, timestamp, coverage, known_cells, total_cells, cov_trace,
  pos_x, pos_y, altitude, frontier_count, nearest_frontier_dist,
  loop_closures, tracking_lost_events, new_cells, note

Usage:
  python3 baseline_random_walk.py
"""

import os
import sys
import math
import time
import csv
import threading
import numpy as np
from collections import deque
from datetime import datetime

import rclpy
from rclpy.executors import MultiThreadedExecutor

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from envs.slam_collector import SLAMDataCollector

# ------------------------------------------------------------------ constants
FLIGHT_ALTITUDE        = -1.2
MAX_STEP_DIST          = 0.5
STEP_DURATION          = 2.0
MAX_STEPS              = 2000
MAX_YAW_CHANGE         = 0.20
WS_MIN_X, WS_MAX_X     = -8.0, 8.0
WS_MIN_Y, WS_MAX_Y     = -5.0, 5.0
LIDAR_COLLISION_DIST   = 0.35
LIDAR_SAFETY_DIST      = 0.5
DEPTH_SAFETY_DIST      = 0.5
MIN_ALTITUDE           = 0.15
ALT_CORRECTION_THRESH  = 0.20
MAX_RESAMPLE_ATTEMPTS  = 8   # How many random directions to try before hovering
MAX_TRACKING_LOST_STEPS = 50
MIN_FRONTIER_DIST      = 0.5


class RandomWalkExplorer:
    """Drone explorer that selects actions uniformly at random."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        if not rclpy.ok():
            rclpy.init()
        self.node = SLAMDataCollector()
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True
        )
        self.spin_thread.start()

        self.prev_yaw              = 0.0
        self.prev_known_cells      = 0
        self.tracking_lost_steps   = 0
        self.total_loop_closures   = 0
        self.total_tracking_lost   = 0
        self.prev_lc_id            = 0
        self.metrics: list[dict]   = []
        self._wait_for_data()

    # ---------------------------------------------------------------- helpers
    def _wait_for_data(self, timeout: float = 30.0) -> bool:
        print("[RandomWalk] Waiting for sensor data …")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (self.node.local_position is not None
                    and self.node.occupancy_grid is not None):
                print("[RandomWalk] Sensor data received.")
                return True
            time.sleep(0.5)
        print("[RandomWalk] WARNING: Timeout waiting for sensor data.")
        return False

    def arm_and_takeoff(self) -> bool:
        target_z   = FLIGHT_ALTITUDE
        target_alt = -target_z
        print(f"[RandomWalk] Arming and taking off to {target_alt:.1f} m …")
        self.node.start_offboard_stream(0.0, 0.0, target_z, yaw=0.0)
        time.sleep(3.0)
        for attempt in range(3):
            self.node.engage_offboard()
            time.sleep(0.3)
            self.node.arm()
            time.sleep(0.5)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self.node.is_armed():
                    print(f"[RandomWalk] Armed (attempt {attempt + 1}).")
                    break
                time.sleep(0.1)
            if self.node.is_armed():
                break
            time.sleep(1.0)
        if not self.node.is_armed():
            print("[RandomWalk] FATAL: Could not arm.")
            return False
        deadline = time.time() + 30.0
        while time.time() < deadline:
            alt = self.node.get_altitude()
            if alt is not None and abs(alt - target_alt) < 0.2:
                print(f"[RandomWalk] Reached {alt:.2f} m")
                break
            self.node.update_setpoint(0.0, 0.0, target_z, yaw=0.0)
            time.sleep(0.1)
        print("[RandomWalk] Stabilising (3 s) …")
        for _ in range(30):
            self.node.update_setpoint(0.0, 0.0, target_z, yaw=0.0)
            time.sleep(0.1)
        self.prev_known_cells = self.node.known_cells_count
        return True

    def _check_safety(self, yaw: float) -> bool:
        if self.node.get_min_lidar_range() < LIDAR_COLLISION_DIST:
            return False
        if self.node.get_lidar_range_in_direction(yaw) < LIDAR_SAFETY_DIST:
            return False
        if self.node.get_depth_min_distance() < DEPTH_SAFETY_DIST:
            return False
        return True

    def _correct_altitude(self):
        alt = self.node.get_altitude()
        if alt is None:
            return
        target_alt = -FLIGHT_ALTITUDE
        if abs(alt - target_alt) > ALT_CORRECTION_THRESH:
            pos = self.node.get_drone_position()
            if pos is not None:
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    self.node.update_setpoint(
                        pos[0], pos[1], FLIGHT_ALTITUDE, yaw=self.prev_yaw
                    )
                    time.sleep(0.2)
                    alt = self.node.get_altitude()
                    if alt is not None and abs(alt - target_alt) <= ALT_CORRECTION_THRESH:
                        break

    # ---------------------------------------------------------------- random step
    def _sample_random_step(self):
        """Sample a random displacement; retry if obstructed."""
        pos = self.node.get_drone_position()
        if pos is None:
            return None

        for _ in range(MAX_RESAMPLE_ATTEMPTS):
            # Sample angle and magnitude uniformly
            angle = self.rng.uniform(-math.pi, math.pi)
            mag   = self.rng.uniform(0.1, MAX_STEP_DIST)
            dx    = math.cos(angle) * mag
            dy    = math.sin(angle) * mag

            next_x = float(np.clip(pos[0] + dx, WS_MIN_X, WS_MAX_X))
            next_y = float(np.clip(pos[1] + dy, WS_MIN_Y, WS_MAX_Y))

            # Rate-limit yaw change
            desired_yaw = angle
            yaw_diff = (desired_yaw - self.prev_yaw + math.pi) % (2 * math.pi) - math.pi
            yaw_diff  = float(np.clip(yaw_diff, -MAX_YAW_CHANGE, MAX_YAW_CHANGE))
            target_yaw = (self.prev_yaw + yaw_diff + math.pi) % (2 * math.pi) - math.pi

            if self._check_safety(target_yaw):
                self.prev_yaw = target_yaw
                return (next_x, next_y, target_yaw)

        # All samples blocked — hover in place
        return (float(pos[0]), float(pos[1]), self.prev_yaw)

    # ---------------------------------------------------------------- frontier helpers (for metrics only)
    def _detect_frontiers(self) -> list:
        """Detect frontier clusters (used only for metric logging)."""
        grid = self.node.occupancy_grid
        if grid is None:
            return []
        h, w = grid.shape
        free_mask    = (grid == 0)
        unknown_mask = (grid == -1)
        unknown_adj  = np.zeros_like(unknown_mask)
        unknown_adj[1:, :]  |= unknown_mask[:-1, :]
        unknown_adj[:-1, :] |= unknown_mask[1:, :]
        unknown_adj[:, 1:]  |= unknown_mask[:, :-1]
        unknown_adj[:, :-1] |= unknown_mask[:, 1:]
        frontier_mask = free_mask & unknown_adj

        labels   = np.zeros((h, w), dtype=np.int32)
        label_id = 0
        clusters = []
        for y in range(h):
            for x in range(w):
                if frontier_mask[y, x] and labels[y, x] == 0:
                    label_id += 1
                    queue = deque([(y, x)])
                    labels[y, x] = label_id
                    cells = []
                    while queue:
                        cy, cx = queue.popleft()
                        cells.append((cx, cy))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if frontier_mask[ny, nx] and labels[ny, nx] == 0:
                                    labels[ny, nx] = label_id
                                    queue.append((ny, nx))
                    if len(cells) >= 3:
                        cx_m = float(np.mean([c[0] for c in cells]))
                        cy_m = float(np.mean([c[1] for c in cells]))
                        clusters.append((cx_m, cy_m, len(cells)))

        gi = self.node.grid_info
        if gi is None:
            return []
        world = []
        for cx_g, cy_g, sz in clusters:
            wx = cx_g * gi.resolution + gi.origin.position.x
            wy = cy_g * gi.resolution + gi.origin.position.y
            world.append((wx, wy, sz))
        return world

    # ---------------------------------------------------------------- logging
    def log_metrics(self, step: int, note: str = "") -> dict:
        pos     = self.node.get_drone_position()
        alt     = self.node.get_altitude()
        cov     = self.node.get_covariance_trace()
        known   = self.node.known_cells_count
        total   = max(self.node.total_cells, 1)
        coverage  = known / total
        new_cells = max(0, known - self.prev_known_cells)
        self.prev_known_cells = known

        lc_id = self.node.loop_closure_id
        if lc_id != 0 and lc_id != self.prev_lc_id:
            self.total_loop_closures += 1
        self.prev_lc_id = lc_id

        clusters = self._detect_frontiers()
        nearest_dist = float("inf")
        if pos is not None:
            for fx, fy, _ in clusters:
                d = math.sqrt((fx - pos[0]) ** 2 + (fy - pos[1]) ** 2)
                nearest_dist = min(nearest_dist, d)

        row = {
            "step":                  step,
            "timestamp":             time.time(),
            "coverage":              round(coverage, 4),
            "known_cells":           known,
            "total_cells":           total,
            "cov_trace":             round(cov, 6),
            "pos_x":                 round(pos[0], 3) if pos is not None else 0,
            "pos_y":                 round(pos[1], 3) if pos is not None else 0,
            "altitude":              round(alt, 3) if alt is not None else 0,
            "frontier_count":        len(clusters),
            "nearest_frontier_dist": round(nearest_dist, 3) if nearest_dist < float("inf") else -1,
            "loop_closures":         self.total_loop_closures,
            "tracking_lost_events":  self.total_tracking_lost,
            "new_cells":             new_cells,
            "note":                  note,
        }
        self.metrics.append(row)
        return row

    def save_metrics(self, filepath: str):
        if not self.metrics:
            return
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics)
        print(f"[RandomWalk] Metrics saved → {filepath} ({len(self.metrics)} rows)")

    # ---------------------------------------------------------------- main loop
    def run(self):
        if not self.arm_and_takeoff():
            return

        print("[RandomWalk] Starting random-walk exploration …")
        start_time = time.time()
        step = 0
        no_data_count = 0

        try:
            for step in range(1, MAX_STEPS + 1):
                self._correct_altitude()

                # Tracking-loss guard
                if self.node.is_tracking_lost():
                    self.tracking_lost_steps   += 1
                    self.total_tracking_lost   += 1
                    pos = self.node.get_drone_position()
                    if pos is not None:
                        self.node.update_setpoint(
                            pos[0], pos[1], FLIGHT_ALTITUDE, yaw=self.prev_yaw
                        )
                    time.sleep(STEP_DURATION)
                    if self.tracking_lost_steps > MAX_TRACKING_LOST_STEPS:
                        print(f"[Step {step}] Tracking lost too long — stopping.")
                        self.log_metrics(step, "tracking_lost_timeout")
                        break
                    self.log_metrics(step, "tracking_lost")
                    continue
                else:
                    self.tracking_lost_steps = 0

                result = self._sample_random_step()
                if result is None:
                    no_data_count += 1
                    time.sleep(STEP_DURATION)
                    if no_data_count > 20:
                        print("[RandomWalk] No position data for 20 steps — aborting.")
                        break
                    continue
                no_data_count = 0

                nx, ny, yaw = result
                self.node.update_setpoint(nx, ny, FLIGHT_ALTITUDE, yaw=yaw)
                time.sleep(STEP_DURATION)

                row = self.log_metrics(step)
                alt = self.node.get_altitude()
                if alt is not None and alt < MIN_ALTITUDE:
                    print(f"[Step {step}] Crashed — alt={alt:.2f} m")
                    break
                if row["coverage"] > 0.95:
                    print(f"[Step {step}] Coverage target reached: {row['coverage']:.1%}")
                    break
                if step % 50 == 0:
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                    pos = self.node.get_drone_position()
                    ps  = f"({pos[0]:.1f},{pos[1]:.1f})" if pos is not None else "?"
                    print(
                        f"[Step {step}/{MAX_STEPS}] "
                        f"Cov:{row['coverage']:.1%} "
                        f"Frt:{row['frontier_count']} "
                        f"Pos:{ps} "
                        f"CovTr:{row['cov_trace']:.4f} "
                        f"New:{row['new_cells']} "
                        f"{elapsed}"
                    )
        except KeyboardInterrupt:
            print("\n[RandomWalk] Interrupted.")

        elapsed = time.time() - start_time
        final   = self.log_metrics(step if step > 0 else 0, "final")
        print("\n" + "=" * 64)
        print("  RANDOM WALK RESULTS")
        print("=" * 64)
        print(f"  Steps:    {len(self.metrics)}")
        print(f"  Coverage: {final['coverage']:.1%}")
        print(f"  Cells:    {final['known_cells']}")
        print(f"  CovTrace: {final['cov_trace']:.4f}")
        print(f"  LoopClos: {final['loop_closures']}")
        print(f"  TrkLost:  {final['tracking_lost_events']}")
        print(f"  Time:     {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
        print("=" * 64)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
        csv_path = os.path.join(
            PROJECT_ROOT, "logs", f"baseline_random_walk_{timestamp}.csv"
        )
        self.save_metrics(csv_path)

    def shutdown(self):
        try:
            self.node.stop_offboard_stream()
            self.node.land()
            time.sleep(5.0)
            self.node.disarm()
        except Exception as e:
            print(f"[RandomWalk] Shutdown error: {e}")
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def main():
    explorer = RandomWalkExplorer(seed=42)
    try:
        explorer.run()
    finally:
        explorer.shutdown()


if __name__ == "__main__":
    main()
