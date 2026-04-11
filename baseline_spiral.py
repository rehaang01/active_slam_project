#!/usr/bin/env python3
"""
Baseline: Spiral / Lawnmower Systematic Coverage

Strategy:
  The drone sweeps the entire workspace in a boustrophedon (back-and-forth)
  lawnmower pattern at a fixed lateral spacing, then optionally tightens into a
  closing inward spiral over any uncovered pockets.

  Phase 1 — Lawnmower:
    Pre-plan an axis-aligned grid of waypoints that covers the entire workspace
    with LANE_SPACING between rows. Move NED-frame along each row, alternating
    direction.  Stop when all rows have been visited OR a coverage target is met.

  Phase 2 — Spiral closing:
    After the lawnmower, if coverage < SPIRAL_TRIGGER_COVERAGE, execute an
    outward-then-inward Archimedean spiral from the current position to pick up
    remaining cells.

This baseline requires NO map information at run-time (it is purely geometric),
making it a strong test of whether the RL policy actually learns something
beyond systematic coverage.

Metrics logged — identical schema to baseline_frontier.py.

Usage:
  python3 baseline_spiral.py
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
WS_MIN_X, WS_MAX_X     = -8.0,  8.0
WS_MIN_Y, WS_MAX_Y     = -5.0,  5.0
LIDAR_COLLISION_DIST   = 0.35
LIDAR_SAFETY_DIST      = 0.5
DEPTH_SAFETY_DIST      = 0.5
MIN_ALTITUDE           = 0.15
ALT_CORRECTION_THRESH  = 0.20
MAX_TRACKING_LOST_STEPS = 50
MIN_FRONTIER_DIST      = 0.5

# Lawnmower / spiral parameters
LANE_SPACING           = 1.0    # Meters between lawnmower rows (≈ sensor FOV width)
WAYPOINT_REACH_DIST    = 0.35   # Consider waypoint reached when within this distance
SPIRAL_TRIGGER_COV     = 0.85   # Switch from lawnmower to spiral if still below this
SPIRAL_STEP_SIZE       = 0.3    # Angular increment per spiral step (radians)
SPIRAL_RADIUS_GROWTH   = 0.08   # Meters added to radius per step
SPIRAL_MAX_RADIUS      = 6.0    # Max spiral radius (stops before workspace edge)
OBSTACLE_SLIDE_OFFSETS = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]


def _plan_lawnmower() -> list[tuple[float, float]]:
    """Generate an ordered list of (x, y) waypoints for boustrophedon coverage."""
    waypoints: list[tuple[float, float]] = []
    y_vals = np.arange(WS_MIN_Y, WS_MAX_Y + LANE_SPACING, LANE_SPACING)
    for i, y in enumerate(y_vals):
        xs = np.arange(WS_MIN_X, WS_MAX_X + LANE_SPACING, LANE_SPACING)
        if i % 2 == 1:
            xs = xs[::-1]  # Alternate row direction
        for x in xs:
            waypoints.append((float(x), float(y)))
    return waypoints


class SpiralExplorer:
    """Drone explorer using pre-planned lawnmower + spiral coverage."""

    def __init__(self):
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
        self.lawnmower_waypoints   = _plan_lawnmower()
        self.waypoint_idx          = 0
        self.spiral_angle          = 0.0
        self.spiral_radius         = LANE_SPACING / 2
        self.phase                 = "lawnmower"   # or "spiral"
        self._wait_for_data()

    # ---------------------------------------------------------------- helpers
    def _wait_for_data(self, timeout: float = 30.0) -> bool:
        print("[Spiral] Waiting for sensor data …")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (self.node.local_position is not None
                    and self.node.occupancy_grid is not None):
                print("[Spiral] Sensor data received.")
                return True
            time.sleep(0.5)
        print("[Spiral] WARNING: Timeout.")
        return False

    def arm_and_takeoff(self) -> bool:
        target_z   = FLIGHT_ALTITUDE
        target_alt = -target_z
        print(f"[Spiral] Arming and taking off to {target_alt:.1f} m …")
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
                    print(f"[Spiral] Armed (attempt {attempt + 1}).")
                    break
                time.sleep(0.1)
            if self.node.is_armed():
                break
            time.sleep(1.0)
        if not self.node.is_armed():
            print("[Spiral] FATAL: Could not arm.")
            return False
        deadline = time.time() + 30.0
        while time.time() < deadline:
            alt = self.node.get_altitude()
            if alt is not None and abs(alt - target_alt) < 0.2:
                print(f"[Spiral] Reached {alt:.2f} m")
                break
            self.node.update_setpoint(0.0, 0.0, target_z, yaw=0.0)
            time.sleep(0.1)
        print("[Spiral] Stabilising (3 s) …")
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
        if abs(alt - (-FLIGHT_ALTITUDE)) > ALT_CORRECTION_THRESH:
            pos = self.node.get_drone_position()
            if pos is not None:
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    self.node.update_setpoint(
                        pos[0], pos[1], FLIGHT_ALTITUDE, yaw=self.prev_yaw
                    )
                    time.sleep(0.2)
                    alt = self.node.get_altitude()
                    if alt is not None and abs(alt - (-FLIGHT_ALTITUDE)) <= ALT_CORRECTION_THRESH:
                        break

    # ---------------------------------------------------------------- motion
    def _step_toward(self, target_x: float, target_y: float):
        """Move MAX_STEP_DIST toward (target_x, target_y). Returns (nx, ny, yaw) or None."""
        pos = self.node.get_drone_position()
        if pos is None:
            return None
        dx   = target_x - pos[0]
        dy   = target_y - pos[1]
        dist = math.hypot(dx, dy)
        if dist > MAX_STEP_DIST:
            dx = dx / dist * MAX_STEP_DIST
            dy = dy / dist * MAX_STEP_DIST
        nx = float(np.clip(pos[0] + dx, WS_MIN_X, WS_MAX_X))
        ny = float(np.clip(pos[1] + dy, WS_MIN_Y, WS_MAX_Y))

        desired_yaw  = math.atan2(dy, dx)
        yaw_diff     = (desired_yaw - self.prev_yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_diff     = float(np.clip(yaw_diff, -MAX_YAW_CHANGE, MAX_YAW_CHANGE))
        target_yaw   = (self.prev_yaw + yaw_diff + math.pi) % (2 * math.pi) - math.pi

        if self._check_safety(target_yaw):
            self.prev_yaw = target_yaw
            return (nx, ny, target_yaw)

        # Obstacle: try slide offsets
        for offset in OBSTACLE_SLIDE_OFFSETS:
            slide_yaw = target_yaw + offset
            if self._check_safety(slide_yaw):
                snx = float(np.clip(pos[0] + math.cos(slide_yaw) * MAX_STEP_DIST * 0.3, WS_MIN_X, WS_MAX_X))
                sny = float(np.clip(pos[1] + math.sin(slide_yaw) * MAX_STEP_DIST * 0.3, WS_MIN_Y, WS_MAX_Y))
                self.prev_yaw = slide_yaw
                return (snx, sny, slide_yaw)

        # Fully blocked — hover
        return (float(pos[0]), float(pos[1]), self.prev_yaw)

    def _next_lawnmower_step(self):
        """Return next step toward current lawnmower waypoint, advance when reached."""
        pos = self.node.get_drone_position()
        if pos is None:
            return None

        # Advance waypoint index while drone is close enough
        while self.waypoint_idx < len(self.lawnmower_waypoints):
            wx, wy = self.lawnmower_waypoints[self.waypoint_idx]
            dist   = math.hypot(wx - pos[0], wy - pos[1])
            if dist < WAYPOINT_REACH_DIST:
                self.waypoint_idx += 1
            else:
                break

        if self.waypoint_idx >= len(self.lawnmower_waypoints):
            return None  # All waypoints visited

        wx, wy = self.lawnmower_waypoints[self.waypoint_idx]
        return self._step_toward(wx, wy)

    def _next_spiral_step(self):
        """Return next step on an outward Archimedean spiral."""
        pos = self.node.get_drone_position()
        if pos is None:
            return None

        self.spiral_angle  += SPIRAL_STEP_SIZE
        self.spiral_radius += SPIRAL_RADIUS_GROWTH
        if self.spiral_radius > SPIRAL_MAX_RADIUS:
            # Reset to tight spiral at centre of workspace
            self.spiral_radius = LANE_SPACING / 2
            self.spiral_angle  = 0.0

        cx = 0.0  # centre of workspace
        cy = 0.0
        tx = float(np.clip(cx + self.spiral_radius * math.cos(self.spiral_angle), WS_MIN_X, WS_MAX_X))
        ty = float(np.clip(cy + self.spiral_radius * math.sin(self.spiral_angle), WS_MIN_Y, WS_MAX_Y))
        return self._step_toward(tx, ty)

    # ---------------------------------------------------------------- frontiers (metrics only)
    def _detect_frontiers(self) -> list:
        grid = self.node.occupancy_grid
        if grid is None:
            return []
        h, w         = grid.shape
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
                            ny, nx_ = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx_ < w:
                                if frontier_mask[ny, nx_] and labels[ny, nx_] == 0:
                                    labels[ny, nx_] = label_id
                                    queue.append((ny, nx_))
                    if len(cells) >= 3:
                        clusters.append((
                            float(np.mean([c[0] for c in cells])),
                            float(np.mean([c[1] for c in cells])),
                            len(cells)
                        ))
        gi = self.node.grid_info
        if gi is None:
            return []
        return [
            (cx * gi.resolution + gi.origin.position.x,
             cy * gi.resolution + gi.origin.position.y, sz)
            for cx, cy, sz in clusters
        ]

    # ---------------------------------------------------------------- logging
    def log_metrics(self, step: int, note: str = "") -> dict:
        pos      = self.node.get_drone_position()
        alt      = self.node.get_altitude()
        cov      = self.node.get_covariance_trace()
        known    = self.node.known_cells_count
        total    = max(self.node.total_cells, 1)
        coverage = known / total
        new_cells = max(0, known - self.prev_known_cells)
        self.prev_known_cells = known

        lc_id = self.node.loop_closure_id
        if lc_id != 0 and lc_id != self.prev_lc_id:
            self.total_loop_closures += 1
        self.prev_lc_id = lc_id

        clusters     = self._detect_frontiers()
        nearest_dist = float("inf")
        if pos is not None:
            for fx, fy, _ in clusters:
                nearest_dist = min(nearest_dist, math.hypot(fx - pos[0], fy - pos[1]))

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
        print(f"[Spiral] Metrics saved → {filepath} ({len(self.metrics)} rows)")

    # ---------------------------------------------------------------- main loop
    def run(self):
        if not self.arm_and_takeoff():
            return

        print(f"[Spiral] Starting — {len(self.lawnmower_waypoints)} lawnmower waypoints planned.")
        start_time = time.time()
        step = 0

        try:
            for step in range(1, MAX_STEPS + 1):
                self._correct_altitude()

                # Tracking-loss guard
                if self.node.is_tracking_lost():
                    self.tracking_lost_steps += 1
                    self.total_tracking_lost += 1
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

                # Decide phase
                known    = self.node.known_cells_count
                total    = max(self.node.total_cells, 1)
                coverage = known / total
                if self.phase == "lawnmower" and self.waypoint_idx >= len(self.lawnmower_waypoints):
                    if coverage < SPIRAL_TRIGGER_COV:
                        self.phase = "spiral"
                        print(f"[Step {step}] Lawnmower complete (cov={coverage:.1%}). Switching to spiral.")
                    else:
                        print(f"[Step {step}] Coverage target reached in lawnmower phase.")
                        break

                if self.phase == "lawnmower":
                    result = self._next_lawnmower_step()
                    note   = f"lawn_wp{self.waypoint_idx}"
                else:
                    result = self._next_spiral_step()
                    note   = f"spiral_r{self.spiral_radius:.2f}"

                if result is None:
                    time.sleep(STEP_DURATION)
                    self.log_metrics(step, "no_result")
                    continue

                nx, ny, yaw = result
                self.node.update_setpoint(nx, ny, FLIGHT_ALTITUDE, yaw=yaw)
                time.sleep(STEP_DURATION)

                row = self.log_metrics(step, note)
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
                        f"Phase:{self.phase} "
                        f"Cov:{row['coverage']:.1%} "
                        f"Frt:{row['frontier_count']} "
                        f"Pos:{ps} "
                        f"CovTr:{row['cov_trace']:.4f} "
                        f"{elapsed}"
                    )
        except KeyboardInterrupt:
            print("\n[Spiral] Interrupted.")

        elapsed = time.time() - start_time
        final   = self.log_metrics(step if step > 0 else 0, "final")
        print("\n" + "=" * 64)
        print("  SPIRAL / LAWNMOWER RESULTS")
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
            PROJECT_ROOT, "logs", f"baseline_spiral_{timestamp}.csv"
        )
        self.save_metrics(csv_path)

    def shutdown(self):
        try:
            self.node.stop_offboard_stream()
            self.node.land()
            time.sleep(5.0)
            self.node.disarm()
        except Exception as e:
            print(f"[Spiral] Shutdown error: {e}")
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def main():
    explorer = SpiralExplorer()
    try:
        explorer.run()
    finally:
        explorer.shutdown()


if __name__ == "__main__":
    main()
