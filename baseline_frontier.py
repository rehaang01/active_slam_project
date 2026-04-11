#!/usr/bin/env python3
"""
Baseline: Nearest-Frontier Exploration (Classical Algorithm) — v2 FIXED

Fixes from v1:
- Always sleep STEP_DURATION between steps (was skipping sleep when target too close)
- Filter out frontiers closer than 0.5m (they're under the drone, not useful)
- If no distant frontiers, do expanding search pattern instead of hovering
- Added debug logging to show what the drone is actually doing
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

FLIGHT_ALTITUDE = -1.2
MAX_STEP_DIST = 0.5
STEP_DURATION = 2.0
MAX_STEPS = 2000
MAX_YAW_CHANGE = 0.20
WS_MIN_X, WS_MAX_X = -8.0, 8.0
WS_MIN_Y, WS_MAX_Y = -5.0, 5.0
LIDAR_COLLISION_DIST = 0.35
LIDAR_SAFETY_DIST = 0.5
DEPTH_SAFETY_DIST = 0.5
MIN_ALTITUDE = 0.15
ALT_CORRECTION_THRESHOLD = 0.20
MIN_FRONTIER_DIST = 0.5
MAX_TRACKING_LOST_STEPS = 50


class FrontierExplorer:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = SLAMDataCollector()
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.prev_yaw = 0.0  # will be updated to the live PX4 heading after takeoff
        self.prev_known_cells = 0
        self.tracking_lost_steps = 0
        self.total_loop_closures = 0
        self.total_tracking_lost_events = 0
        self.prev_lc_id = 0
        self.metrics = []
        self.search_angle = 0.0
        self._wait_for_data()

    def _wait_for_data(self, timeout=30.0):
        print("[Baseline] Waiting for sensor data...")
        start = time.time()
        while time.time() - start < timeout:
            if self.node.local_position is not None and self.node.occupancy_grid is not None:
                print("[Baseline] Sensor data received.")
                return True
            time.sleep(0.5)
        print("[Baseline] WARNING: Timeout.")
        return False

    def arm_and_takeoff(self):
        target_z = FLIGHT_ALTITUDE
        target_alt = -target_z
        print(f"[Baseline] Arming and taking off to {target_alt:.1f}m...")
        self.node.start_offboard_stream(0.0, 0.0, target_z, yaw=0.0)
        time.sleep(3.0)
        for attempt in range(3):
            self.node.engage_offboard()
            time.sleep(0.3)
            self.node.arm()
            time.sleep(0.5)
            start = time.time()
            while time.time() - start < 3.0:
                if self.node.is_armed():
                    print(f"[Baseline] Armed (attempt {attempt + 1}).")
                    break
                time.sleep(0.1)
            if self.node.is_armed():
                break
            time.sleep(1.0)
        if not self.node.is_armed():
            print("[Baseline] FATAL: Could not arm.")
            return False

        # Lock the controller to the current live heading to avoid an immediate yaw snap.
        self.prev_yaw = self.node.get_drone_yaw()

        start = time.time()
        while time.time() - start < 30.0:
            alt = self.node.get_altitude()
            if alt is not None and abs(alt - target_alt) < 0.2:
                print(f"[Baseline] Reached {alt:.2f}m")
                break
            self.node.update_setpoint(0.0, 0.0, target_z, yaw=self.prev_yaw)
            time.sleep(0.1)
        print("[Baseline] Stabilizing (3s)...")
        for _ in range(30):
            self.node.update_setpoint(0.0, 0.0, target_z, yaw=self.prev_yaw)
            time.sleep(0.1)
        alt = self.node.get_altitude()
        print(f"[Baseline] Ready at altitude {alt:.2f}m")
        self.prev_known_cells = self.node.known_cells_count
        return True

    def detect_frontiers(self):
        grid = self.node.occupancy_grid
        if grid is None:
            return []
        h, w = grid.shape
        free_mask = (grid == 0)
        unknown_mask = (grid == -1)
        unknown_adj = np.zeros_like(unknown_mask)
        unknown_adj[1:, :] |= unknown_mask[:-1, :]
        unknown_adj[:-1, :] |= unknown_mask[1:, :]
        unknown_adj[:, 1:] |= unknown_mask[:, :-1]
        unknown_adj[:, :-1] |= unknown_mask[:, 1:]
        frontier_mask = free_mask & unknown_adj
        labels = np.zeros((h, w), dtype=np.int32)
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
                        cx_mean = np.mean([c[0] for c in cells])
                        cy_mean = np.mean([c[1] for c in cells])
                        clusters.append((cx_mean, cy_mean, len(cells)))
        gi = self.node.grid_info
        if gi is None:
            return []
        world_clusters = []
        for cx_grid, cy_grid, size in clusters:
            wx = cx_grid * gi.resolution + gi.origin.position.x
            wy = cy_grid * gi.resolution + gi.origin.position.y
            world_clusters.append((wx, wy, size))
        return world_clusters

    def select_target(self, clusters):
        pos = self.node.get_drone_position()
        if pos is None or len(clusters) == 0:
            return None, float('inf')
        best_score = -1
        best_target = None
        best_dist = float('inf')
        for fx, fy, fsize in clusters:
            dist = math.sqrt((fx - pos[0])**2 + (fy - pos[1])**2)
            if dist < MIN_FRONTIER_DIST:
                continue
            score = fsize / (dist + 0.5)
            if score > best_score:
                best_score = score
                best_target = (fx, fy)
                best_dist = dist
        return best_target, best_dist

    def check_safety(self, target_yaw):
        if self.node.get_min_lidar_range() < LIDAR_COLLISION_DIST:
            return False
        range_ahead = self.node.get_lidar_range_in_direction(target_yaw)
        if range_ahead < LIDAR_SAFETY_DIST:
            return False
        if self.node.get_depth_min_distance() < DEPTH_SAFETY_DIST:
            return False
        return True

    def correct_altitude(self):
        alt = self.node.get_altitude()
        if alt is None:
            return
        target_alt = -FLIGHT_ALTITUDE
        if abs(alt - target_alt) > ALT_CORRECTION_THRESHOLD:
            pos = self.node.get_drone_position()
            if pos is not None:
                start = time.time()
                while time.time() - start < 2.0:
                    self.node.update_setpoint(pos[0], pos[1], FLIGHT_ALTITUDE, yaw=self.prev_yaw)
                    time.sleep(0.2)
                    alt = self.node.get_altitude()
                    if alt is not None and abs(alt - target_alt) <= ALT_CORRECTION_THRESHOLD:
                        break

    def send_step(self, target_x, target_y, target_yaw):
        self.node.update_setpoint(target_x, target_y, FLIGHT_ALTITUDE, yaw=target_yaw)
        time.sleep(STEP_DURATION)

    def compute_step_toward(self, target_x, target_y):
        pos = self.node.get_drone_position()
        if pos is None:
            return None
        dx = target_x - pos[0]
        dy = target_y - pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        if dist > MAX_STEP_DIST:
            dx = (dx / dist) * MAX_STEP_DIST
            dy = (dy / dist) * MAX_STEP_DIST
        next_x = float(np.clip(pos[0] + dx, WS_MIN_X, WS_MAX_X))
        next_y = float(np.clip(pos[1] + dy, WS_MIN_Y, WS_MAX_Y))
        desired_yaw = math.atan2(dy, dx)
        lidar_min = self.node.get_min_lidar_range()
        yaw_limit = 0.05 if lidar_min < 0.6 else MAX_YAW_CHANGE
        yaw_diff = desired_yaw - self.prev_yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        yaw_diff = max(-yaw_limit, min(yaw_limit, yaw_diff))
        target_yaw = self.prev_yaw + yaw_diff
        target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi
        if not self.check_safety(target_yaw):
            for slide_offset in [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]:
                slide_yaw = target_yaw + slide_offset
                if self.check_safety(slide_yaw):
                    next_x = float(np.clip(pos[0] + math.cos(slide_yaw) * MAX_STEP_DIST * 0.3, WS_MIN_X, WS_MAX_X))
                    next_y = float(np.clip(pos[1] + math.sin(slide_yaw) * MAX_STEP_DIST * 0.3, WS_MIN_Y, WS_MAX_Y))
                    target_yaw = slide_yaw
                    self.prev_yaw = target_yaw
                    return (next_x, next_y, target_yaw)
            return (float(pos[0]), float(pos[1]), self.prev_yaw)
        self.prev_yaw = target_yaw
        return (next_x, next_y, target_yaw)

    def do_search_pattern(self):
        pos = self.node.get_drone_position()
        if pos is None:
            time.sleep(STEP_DURATION)
            return
        self.search_angle += 0.4
        radius = 0.4
        target_x = float(np.clip(pos[0] + math.cos(self.search_angle) * radius, WS_MIN_X, WS_MAX_X))
        target_y = float(np.clip(pos[1] + math.sin(self.search_angle) * radius, WS_MIN_Y, WS_MAX_Y))
        result = self.compute_step_toward(target_x, target_y)
        if result:
            self.send_step(*result)
        else:
            self.send_step(float(pos[0]), float(pos[1]), self.prev_yaw)

    def log_metrics(self, step, extra=None):
        pos = self.node.get_drone_position()
        alt = self.node.get_altitude()
        cov = self.node.get_covariance_trace()
        current_known = self.node.known_cells_count
        total = max(self.node.total_cells, 1)
        coverage = current_known / total
        new_cells = max(0, current_known - self.prev_known_cells)
        self.prev_known_cells = current_known
        if self.node.loop_closure_id != 0 and self.node.loop_closure_id != self.prev_lc_id:
            self.total_loop_closures += 1
        self.prev_lc_id = self.node.loop_closure_id
        clusters = self.detect_frontiers()
        nearest_dist = float('inf')
        if pos is not None and len(clusters) > 0:
            for fx, fy, _ in clusters:
                d = math.sqrt((fx - pos[0])**2 + (fy - pos[1])**2)
                nearest_dist = min(nearest_dist, d)
        row = {
            "step": step, "timestamp": time.time(),
            "coverage": round(coverage, 4), "known_cells": current_known,
            "total_cells": total, "cov_trace": round(cov, 6),
            "pos_x": round(pos[0], 3) if pos is not None else 0,
            "pos_y": round(pos[1], 3) if pos is not None else 0,
            "altitude": round(alt, 3) if alt is not None else 0,
            "frontier_count": len(clusters),
            "nearest_frontier_dist": round(nearest_dist, 3) if nearest_dist < float('inf') else -1,
            "loop_closures": self.total_loop_closures,
            "tracking_lost_events": self.total_tracking_lost_events,
            "new_cells": new_cells, "note": extra or "",
        }
        self.metrics.append(row)
        return row

    def save_metrics(self, filepath):
        if len(self.metrics) == 0:
            return
        keys = self.metrics[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.metrics)
        print(f"[Baseline] Metrics saved to {filepath} ({len(self.metrics)} rows)")

    def run(self):
        print("=" * 64)
        print("  Baseline: Nearest-Frontier Exploration v2")
        print(f"  Max steps: {MAX_STEPS} (~{MAX_STEPS * STEP_DURATION / 60:.0f} min)")
        print(f"  Altitude: {-FLIGHT_ALTITUDE:.1f}m | Step: {MAX_STEP_DIST}m/{STEP_DURATION}s")
        print("=" * 64)
        if not self.arm_and_takeoff():
            return
        start_time = time.time()
        no_useful_frontier_count = 0
        try:
            for step in range(1, MAX_STEPS + 1):
                self.correct_altitude()
                if self.node.is_tracking_lost():
                    self.tracking_lost_steps += 1
                    if self.tracking_lost_steps == 1:
                        self.total_tracking_lost_events += 1
                        print(f"[Step {step}] TRACKING LOST")

                    # Do not fly to origin here. That aggressive lateral move can
                    # create a yaw snap and destabilize RTAB-Map/PX4. Hold position
                    # and keep the last good heading instead.
                    pos = self.node.get_drone_position()
                    if pos is not None:
                        self.node.update_setpoint(
                            float(pos[0]), float(pos[1]), FLIGHT_ALTITUDE, yaw=self.prev_yaw
                        )
                    time.sleep(STEP_DURATION)

                    if self.tracking_lost_steps >= MAX_TRACKING_LOST_STEPS:
                        print(f"[Step {step}] Lost too long. Stopping.")
                        self.log_metrics(step, "tracking_lost_timeout")
                        break
                    self.log_metrics(step, "tracking_lost")
                    continue
                else:
                    if self.tracking_lost_steps > 0:
                        print(f"[Step {step}] Tracking RECOVERED after {self.tracking_lost_steps} steps")
                    self.tracking_lost_steps = 0
                clusters = self.detect_frontiers()
                target, target_dist = self.select_target(clusters)
                if target is None:
                    no_useful_frontier_count += 1
                    self.do_search_pattern()
                    note = f"search(x{no_useful_frontier_count})"
                    if no_useful_frontier_count >= 200:
                        print(f"[Step {step}] No useful frontiers for 200 steps. Done.")
                        self.log_metrics(step, "done")
                        break
                else:
                    no_useful_frontier_count = 0
                    result = self.compute_step_toward(target[0], target[1])
                    if result:
                        self.send_step(*result)
                    else:
                        pos = self.node.get_drone_position()
                        if pos is not None:
                            self.send_step(float(pos[0]), float(pos[1]), self.prev_yaw)
                        else:
                            time.sleep(STEP_DURATION)
                    note = f"go({target[0]:.1f},{target[1]:.1f})d={target_dist:.1f}"
                row = self.log_metrics(step, note)
                alt = self.node.get_altitude()
                if alt is not None and alt < MIN_ALTITUDE:
                    print(f"[Step {step}] CRASHED alt={alt:.2f}m")
                    break
                if row["coverage"] > 0.95:
                    print(f"[Step {step}] TARGET: {row['coverage']:.1%}")
                    break
                if step % 25 == 0:
                    elapsed = time.time() - start_time
                    estr = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    pos = self.node.get_drone_position()
                    ps = f"({pos[0]:.1f},{pos[1]:.1f})" if pos is not None else "?"
                    print(f"[Step {step}/{MAX_STEPS}] "
                          f"Cov:{row['coverage']:.1%} "
                          f"Frt:{row['frontier_count']} "
                          f"Pos:{ps} "
                          f"Alt:{row['altitude']:.2f} "
                          f"Cov:{row['cov_trace']:.4f} "
                          f"New:{row['new_cells']} "
                          f"LC:{row['loop_closures']} "
                          f"{estr} | {note[:30]}")
        except KeyboardInterrupt:
            print("\n[Baseline] Interrupted.")
        elapsed = time.time() - start_time
        final = self.log_metrics(step if 'step' in dir() else 0, "final")
        print("\n" + "=" * 64)
        print("  BASELINE RESULTS")
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
        csv_path = os.path.join(PROJECT_ROOT, "logs", f"baseline_frontier_{timestamp}.csv")
        self.save_metrics(csv_path)

    def shutdown(self):
        try:
            self.node.stop_offboard_stream()
            self.node.land()
            time.sleep(5.0)
            self.node.disarm()
        except Exception as e:
            print(f"[Baseline] Shutdown error: {e}")
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def main():
    explorer = FrontierExplorer()
    try:
        explorer.run()
    finally:
        explorer.shutdown()


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Baseline: Nearest-Frontier Exploration (Classical Algorithm) — v2 FIXED

Fixes from v1:
- Always sleep STEP_DURATION between steps (was skipping sleep when target too close)
- Filter out frontiers closer than 0.5m (they're under the drone, not useful)
- If no distant frontiers, do expanding search pattern instead of hovering
- Added debug logging to show what the drone is actually doing
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

FLIGHT_ALTITUDE = -1.2
MAX_STEP_DIST = 0.5
STEP_DURATION = 2.0
MAX_STEPS = 2000
MAX_YAW_CHANGE = 0.20
WS_MIN_X, WS_MAX_X = -8.0, 8.0
WS_MIN_Y, WS_MAX_Y = -5.0, 5.0
LIDAR_COLLISION_DIST = 0.35
LIDAR_SAFETY_DIST = 0.5
DEPTH_SAFETY_DIST = 0.5
MIN_ALTITUDE = 0.15
ALT_CORRECTION_THRESHOLD = 0.20
MIN_FRONTIER_DIST = 0.5
MAX_TRACKING_LOST_STEPS = 50


class FrontierExplorer:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = SLAMDataCollector()
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.prev_yaw = 0.0
        self.prev_known_cells = 0
        self.tracking_lost_steps = 0
        self.total_loop_closures = 0
        self.total_tracking_lost_events = 0
        self.prev_lc_id = 0
        self.metrics = []
        self.search_angle = 0.0
        self._wait_for_data()

    def _wait_for_data(self, timeout=30.0):
        print("[Baseline] Waiting for sensor data...")
        start = time.time()
        while time.time() - start < timeout:
            if self.node.local_position is not None and self.node.occupancy_grid is not None:
                print("[Baseline] Sensor data received.")
                return True
            time.sleep(0.5)
        print("[Baseline] WARNING: Timeout.")
        return False

    def arm_and_takeoff(self):
        target_z = FLIGHT_ALTITUDE
        target_alt = -target_z
        print(f"[Baseline] Arming and taking off to {target_alt:.1f}m...")
        self.node.start_offboard_stream(0.0, 0.0, target_z, yaw=0.0)
        time.sleep(3.0)
        for attempt in range(3):
            self.node.engage_offboard()
            time.sleep(0.3)
            self.node.arm()
            time.sleep(0.5)
            start = time.time()
            while time.time() - start < 3.0:
                if self.node.is_armed():
                    print(f"[Baseline] Armed (attempt {attempt + 1}).")
                    break
                time.sleep(0.1)
            if self.node.is_armed():
                break
            time.sleep(1.0)
        if not self.node.is_armed():
            print("[Baseline] FATAL: Could not arm.")
            return False
        start = time.time()
        while time.time() - start < 30.0:
            alt = self.node.get_altitude()
            if alt is not None and abs(alt - target_alt) < 0.2:
                print(f"[Baseline] Reached {alt:.2f}m")
                break
            self.node.update_setpoint(0.0, 0.0, target_z, yaw=0.0)
            time.sleep(0.1)
        print("[Baseline] Stabilizing (3s)...")
        for _ in range(30):
            self.node.update_setpoint(0.0, 0.0, target_z, yaw=0.0)
            time.sleep(0.1)
        alt = self.node.get_altitude()
        print(f"[Baseline] Ready at altitude {alt:.2f}m")
        self.prev_known_cells = self.node.known_cells_count
        return True

    def detect_frontiers(self):
        grid = self.node.occupancy_grid
        if grid is None:
            return []
        h, w = grid.shape
        free_mask = (grid == 0)
        unknown_mask = (grid == -1)
        unknown_adj = np.zeros_like(unknown_mask)
        unknown_adj[1:, :] |= unknown_mask[:-1, :]
        unknown_adj[:-1, :] |= unknown_mask[1:, :]
        unknown_adj[:, 1:] |= unknown_mask[:, :-1]
        unknown_adj[:, :-1] |= unknown_mask[:, 1:]
        frontier_mask = free_mask & unknown_adj
        labels = np.zeros((h, w), dtype=np.int32)
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
                        cx_mean = np.mean([c[0] for c in cells])
                        cy_mean = np.mean([c[1] for c in cells])
                        clusters.append((cx_mean, cy_mean, len(cells)))
        gi = self.node.grid_info
        if gi is None:
            return []
        world_clusters = []
        for cx_grid, cy_grid, size in clusters:
            wx = cx_grid * gi.resolution + gi.origin.position.x
            wy = cy_grid * gi.resolution + gi.origin.position.y
            world_clusters.append((wx, wy, size))
        return world_clusters

    def select_target(self, clusters):
        pos = self.node.get_drone_position()
        if pos is None or len(clusters) == 0:
            return None, float('inf')
        best_score = -1
        best_target = None
        best_dist = float('inf')
        for fx, fy, fsize in clusters:
            dist = math.sqrt((fx - pos[0])**2 + (fy - pos[1])**2)
            if dist < MIN_FRONTIER_DIST:
                continue
            score = fsize / (dist + 0.5)
            if score > best_score:
                best_score = score
                best_target = (fx, fy)
                best_dist = dist
        return best_target, best_dist

    def check_safety(self, target_yaw):
        if self.node.get_min_lidar_range() < LIDAR_COLLISION_DIST:
            return False
        range_ahead = self.node.get_lidar_range_in_direction(target_yaw)
        if range_ahead < LIDAR_SAFETY_DIST:
            return False
        if self.node.get_depth_min_distance() < DEPTH_SAFETY_DIST:
            return False
        return True

    def correct_altitude(self):
        alt = self.node.get_altitude()
        if alt is None:
            return
        target_alt = -FLIGHT_ALTITUDE
        if abs(alt - target_alt) > ALT_CORRECTION_THRESHOLD:
            pos = self.node.get_drone_position()
            if pos is not None:
                start = time.time()
                while time.time() - start < 2.0:
                    self.node.update_setpoint(pos[0], pos[1], FLIGHT_ALTITUDE, yaw=self.prev_yaw)
                    time.sleep(0.2)
                    alt = self.node.get_altitude()
                    if alt is not None and abs(alt - target_alt) <= ALT_CORRECTION_THRESHOLD:
                        break

    def send_step(self, target_x, target_y, target_yaw):
        self.node.update_setpoint(target_x, target_y, FLIGHT_ALTITUDE, yaw=target_yaw)
        time.sleep(STEP_DURATION)

    def compute_step_toward(self, target_x, target_y):
        pos = self.node.get_drone_position()
        if pos is None:
            return None
        dx = target_x - pos[0]
        dy = target_y - pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        if dist > MAX_STEP_DIST:
            dx = (dx / dist) * MAX_STEP_DIST
            dy = (dy / dist) * MAX_STEP_DIST
        next_x = float(np.clip(pos[0] + dx, WS_MIN_X, WS_MAX_X))
        next_y = float(np.clip(pos[1] + dy, WS_MIN_Y, WS_MAX_Y))
        desired_yaw = math.atan2(dy, dx)
        lidar_min = self.node.get_min_lidar_range()
        yaw_limit = 0.05 if lidar_min < 0.6 else MAX_YAW_CHANGE
        yaw_diff = desired_yaw - self.prev_yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        yaw_diff = max(-yaw_limit, min(yaw_limit, yaw_diff))
        target_yaw = self.prev_yaw + yaw_diff
        target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi
        if not self.check_safety(target_yaw):
            for slide_offset in [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]:
                slide_yaw = target_yaw + slide_offset
                if self.check_safety(slide_yaw):
                    next_x = float(np.clip(pos[0] + math.cos(slide_yaw) * MAX_STEP_DIST * 0.3, WS_MIN_X, WS_MAX_X))
                    next_y = float(np.clip(pos[1] + math.sin(slide_yaw) * MAX_STEP_DIST * 0.3, WS_MIN_Y, WS_MAX_Y))
                    target_yaw = slide_yaw
                    self.prev_yaw = target_yaw
                    return (next_x, next_y, target_yaw)
            return (float(pos[0]), float(pos[1]), self.prev_yaw)
        self.prev_yaw = target_yaw
        return (next_x, next_y, target_yaw)

    def do_search_pattern(self):
        pos = self.node.get_drone_position()
        if pos is None:
            time.sleep(STEP_DURATION)
            return
        self.search_angle += 0.4
        radius = 0.4
        target_x = float(np.clip(pos[0] + math.cos(self.search_angle) * radius, WS_MIN_X, WS_MAX_X))
        target_y = float(np.clip(pos[1] + math.sin(self.search_angle) * radius, WS_MIN_Y, WS_MAX_Y))
        result = self.compute_step_toward(target_x, target_y)
        if result:
            self.send_step(*result)
        else:
            self.send_step(float(pos[0]), float(pos[1]), self.prev_yaw)

    def log_metrics(self, step, extra=None):
        pos = self.node.get_drone_position()
        alt = self.node.get_altitude()
        cov = self.node.get_covariance_trace()
        current_known = self.node.known_cells_count
        total = max(self.node.total_cells, 1)
        coverage = current_known / total
        new_cells = max(0, current_known - self.prev_known_cells)
        self.prev_known_cells = current_known
        if self.node.loop_closure_id != 0 and self.node.loop_closure_id != self.prev_lc_id:
            self.total_loop_closures += 1
        self.prev_lc_id = self.node.loop_closure_id
        clusters = self.detect_frontiers()
        nearest_dist = float('inf')
        if pos is not None and len(clusters) > 0:
            for fx, fy, _ in clusters:
                d = math.sqrt((fx - pos[0])**2 + (fy - pos[1])**2)
                nearest_dist = min(nearest_dist, d)
        row = {
            "step": step, "timestamp": time.time(),
            "coverage": round(coverage, 4), "known_cells": current_known,
            "total_cells": total, "cov_trace": round(cov, 6),
            "pos_x": round(pos[0], 3) if pos is not None else 0,
            "pos_y": round(pos[1], 3) if pos is not None else 0,
            "altitude": round(alt, 3) if alt is not None else 0,
            "frontier_count": len(clusters),
            "nearest_frontier_dist": round(nearest_dist, 3) if nearest_dist < float('inf') else -1,
            "loop_closures": self.total_loop_closures,
            "tracking_lost_events": self.total_tracking_lost_events,
            "new_cells": new_cells, "note": extra or "",
        }
        self.metrics.append(row)
        return row

    def save_metrics(self, filepath):
        if len(self.metrics) == 0:
            return
        keys = self.metrics[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.metrics)
        print(f"[Baseline] Metrics saved to {filepath} ({len(self.metrics)} rows)")

    def run(self):
        print("=" * 64)
        print("  Baseline: Nearest-Frontier Exploration v2")
        print(f"  Max steps: {MAX_STEPS} (~{MAX_STEPS * STEP_DURATION / 60:.0f} min)")
        print(f"  Altitude: {-FLIGHT_ALTITUDE:.1f}m | Step: {MAX_STEP_DIST}m/{STEP_DURATION}s")
        print("=" * 64)
        if not self.arm_and_takeoff():
            return
        start_time = time.time()
        no_useful_frontier_count = 0
        try:
            for step in range(1, MAX_STEPS + 1):
                self.correct_altitude()
                if self.node.is_tracking_lost():
                    self.tracking_lost_steps += 1
                    if self.tracking_lost_steps == 1:
                        self.total_tracking_lost_events += 1
                        print(f"[Step {step}] TRACKING LOST")
                    pos = self.node.get_drone_position()
                    if pos is not None:
                        yaw = math.atan2(-pos[1], -pos[0])
                        self.send_step(0.0, 0.0, yaw)
                    else:
                        time.sleep(STEP_DURATION)
                    if self.tracking_lost_steps >= MAX_TRACKING_LOST_STEPS:
                        print(f"[Step {step}] Lost too long. Stopping.")
                        self.log_metrics(step, "tracking_lost_timeout")
                        break
                    self.log_metrics(step, "tracking_lost")
                    continue
                else:
                    if self.tracking_lost_steps > 0:
                        print(f"[Step {step}] Tracking RECOVERED after {self.tracking_lost_steps} steps")
                    self.tracking_lost_steps = 0
                clusters = self.detect_frontiers()
                target, target_dist = self.select_target(clusters)
                if target is None:
                    no_useful_frontier_count += 1
                    self.do_search_pattern()
                    note = f"search(x{no_useful_frontier_count})"
                    if no_useful_frontier_count >= 200:
                        print(f"[Step {step}] No useful frontiers for 200 steps. Done.")
                        self.log_metrics(step, "done")
                        break
                else:
                    no_useful_frontier_count = 0
                    result = self.compute_step_toward(target[0], target[1])
                    if result:
                        self.send_step(*result)
                    else:
                        pos = self.node.get_drone_position()
                        if pos is not None:
                            self.send_step(float(pos[0]), float(pos[1]), self.prev_yaw)
                        else:
                            time.sleep(STEP_DURATION)
                    note = f"go({target[0]:.1f},{target[1]:.1f})d={target_dist:.1f}"
                row = self.log_metrics(step, note)
                alt = self.node.get_altitude()
                if alt is not None and alt < MIN_ALTITUDE:
                    print(f"[Step {step}] CRASHED alt={alt:.2f}m")
                    break
                if row["coverage"] > 0.95:
                    print(f"[Step {step}] TARGET: {row['coverage']:.1%}")
                    break
                if step % 25 == 0:
                    elapsed = time.time() - start_time
                    estr = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    pos = self.node.get_drone_position()
                    ps = f"({pos[0]:.1f},{pos[1]:.1f})" if pos is not None else "?"
                    print(f"[Step {step}/{MAX_STEPS}] "
                          f"Cov:{row['coverage']:.1%} "
                          f"Frt:{row['frontier_count']} "
                          f"Pos:{ps} "
                          f"Alt:{row['altitude']:.2f} "
                          f"Cov:{row['cov_trace']:.4f} "
                          f"New:{row['new_cells']} "
                          f"LC:{row['loop_closures']} "
                          f"{estr} | {note[:30]}")
        except KeyboardInterrupt:
            print("\n[Baseline] Interrupted.")
        elapsed = time.time() - start_time
        final = self.log_metrics(step if 'step' in dir() else 0, "final")
        print("\n" + "=" * 64)
        print("  BASELINE RESULTS")
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
        csv_path = os.path.join(PROJECT_ROOT, "logs", f"baseline_frontier_{timestamp}.csv")
        self.save_metrics(csv_path)

    def shutdown(self):
        try:
            self.node.stop_offboard_stream()
            self.node.land()
            time.sleep(5.0)
            self.node.disarm()
        except Exception as e:
            print(f"[Baseline] Shutdown error: {e}")
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def main():
    explorer = FrontierExplorer()
    try:
        explorer.run()
    finally:
        explorer.shutdown()


if __name__ == "__main__":
    main()