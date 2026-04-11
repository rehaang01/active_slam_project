#!/usr/bin/env python3
"""
Baseline: Information-Theoretic Potential Field (ITPF) Exploration

Strategy:
  Combines classical artificial potential fields with an information-gain
  objective derived from the occupancy map.

  At every step the agent computes a composite potential for every candidate
  heading direction:

      U(θ) = w_info  * InfoGain(θ)        [attractive: unexplored cells ahead]
           - w_cov   * CovPenalty(θ)       [repulsive: high SLAM covariance]
           - w_obs   * ObstaclePenalty(θ)  [repulsive: close obstacles]
           + w_lc    * LoopClosurePotential(θ)  [attractive: revisit loop-able areas]

  The direction maximising U(θ) is chosen, and the drone steps MAX_STEP_DIST
  in that direction.

  This is a stronger baseline than random walk or simple frontier, because it
  simultaneously optimises map coverage and SLAM quality — the same dual
  objective the RL agent is trained for — using a hand-crafted potential.

  Key differences vs. the RL agent:
  * No memory (no LSTM state): each step is decided independently
  * No learned representations: uses raw grid and scalar heuristics
  * No reward shaping or training: weights are hand-tuned

Metrics logged — identical schema to baseline_frontier.py.

Usage:
  python3 baseline_potential_field.py
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

# Potential field weights
W_INFO       = 3.0   # Weight for information gain (unknown cells ahead)
W_COV        = 1.5   # Weight for covariance penalty (avoid high-uncertainty headings)
W_OBS        = 5.0   # Weight for obstacle repulsion
W_LC         = 1.0   # Weight for loop-closure potential (revisit seen areas)
W_FRONTIER   = 2.0   # Weight for frontier distance potential

# Scan parameters for potential evaluation
N_CANDIDATE_HEADINGS = 16       # Number of heading candidates to evaluate
INFO_RAY_LEN_M       = 3.0      # How far ahead (meters) to cast the info ray
LC_REVISIT_RADIUS    = 2.0      # Look for visited cells within this radius
COV_THRESHOLD_HIGH   = 0.8      # Above this cov_trace, apply stronger repulsion
MAX_STUCK_STEPS      = 30       # If new_cells == 0 for this long, perturb direction


class PotentialFieldExplorer:
    """Information-theoretic potential field drone explorer."""

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
        self.stuck_steps           = 0
        self.perturb_angle         = 0.0   # Used when stuck
        self._wait_for_data()

    # ---------------------------------------------------------------- helpers
    def _wait_for_data(self, timeout: float = 30.0) -> bool:
        print("[ITPF] Waiting for sensor data …")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (self.node.local_position is not None
                    and self.node.occupancy_grid is not None):
                print("[ITPF] Sensor data received.")
                return True
            time.sleep(0.5)
        print("[ITPF] WARNING: Timeout.")
        return False

    def arm_and_takeoff(self) -> bool:
        target_z   = FLIGHT_ALTITUDE
        target_alt = -target_z
        print(f"[ITPF] Arming and taking off to {target_alt:.1f} m …")
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
                    print(f"[ITPF] Armed (attempt {attempt + 1}).")
                    break
                time.sleep(0.1)
            if self.node.is_armed():
                break
            time.sleep(1.0)
        if not self.node.is_armed():
            print("[ITPF] FATAL: Could not arm.")
            return False
        deadline = time.time() + 30.0
        while time.time() < deadline:
            alt = self.node.get_altitude()
            if alt is not None and abs(alt - target_alt) < 0.2:
                print(f"[ITPF] Reached {alt:.2f} m")
                break
            self.node.update_setpoint(0.0, 0.0, target_z, yaw=0.0)
            time.sleep(0.1)
        print("[ITPF] Stabilising (3 s) …")
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

    # ---------------------------------------------------------------- potential field
    def _ray_info_gain(self, pos_x: float, pos_y: float,
                       heading: float, grid: np.ndarray, gi) -> float:
        """Count unknown cells along a ray in the given heading direction."""
        if gi is None or grid is None:
            return 0.0
        res = gi.resolution
        ox  = gi.origin.position.x
        oy  = gi.origin.position.y
        h, w = grid.shape
        unknown_count = 0
        steps = int(INFO_RAY_LEN_M / res)
        for i in range(1, steps + 1):
            rx = pos_x + math.cos(heading) * i * res
            ry = pos_y + math.sin(heading) * i * res
            gx = int((rx - ox) / res)
            gy = int((ry - oy) / res)
            if 0 <= gx < w and 0 <= gy < h:
                if grid[gy, gx] == -1:   # Unknown
                    unknown_count += 1
                elif grid[gy, gx] > 50:  # Occupied — ray blocked
                    break
        return float(unknown_count)

    def _frontier_distance_potential(self, pos_x: float, pos_y: float,
                                     heading: float, clusters: list) -> float:
        """Potential based on how well this heading points toward nearest frontier."""
        if not clusters:
            return 0.0
        best = 0.0
        for fx, fy, fsz in clusters:
            dir_to_f = math.atan2(fy - pos_y, fx - pos_x)
            ang_diff = abs((dir_to_f - heading + math.pi) % (2 * math.pi) - math.pi)
            dist     = math.hypot(fx - pos_x, fy - pos_y)
            # Alignment bonus: maximum when heading exactly toward frontier
            alignment = math.cos(ang_diff)  # [-1, 1]
            score = (fsz / (dist + 1.0)) * max(0.0, alignment)
            best = max(best, score)
        return best

    def _loop_closure_potential(self, pos_x: float, pos_y: float,
                                heading: float, grid: np.ndarray, gi) -> float:
        """Potential for loop closure: reward headings toward already-visited areas
        (occupied cells within revisit radius), which helps RTAB-Map close loops."""
        if gi is None or grid is None:
            return 0.0
        res = gi.resolution
        ox  = gi.origin.position.x
        oy  = gi.origin.position.y
        h, w = grid.shape
        visited_score = 0.0
        radius_cells  = int(LC_REVISIT_RADIUS / res)
        gx_c = int((pos_x - ox) / res)
        gy_c = int((pos_y - oy) / res)
        for gy in range(max(0, gy_c - radius_cells), min(h, gy_c + radius_cells)):
            for gx in range(max(0, gx_c - radius_cells), min(w, gx_c + radius_cells)):
                if grid[gy, gx] == 0:  # Free (visited)
                    wx = gx * res + ox
                    wy = gy * res + oy
                    dir_to = math.atan2(wy - pos_y, wx - pos_x)
                    ang_diff = abs((dir_to - heading + math.pi) % (2 * math.pi) - math.pi)
                    if ang_diff < math.pi / 4:  # Within 45°
                        visited_score += 1.0
        return visited_score / max(1.0, (2 * radius_cells) ** 2)

    def _obstacle_penalty(self, heading: float) -> float:
        """Repulsive potential from LiDAR reading in given direction."""
        dist = self.node.get_lidar_range_in_direction(heading)
        if dist < LIDAR_SAFETY_DIST:
            return (LIDAR_SAFETY_DIST - dist) / LIDAR_SAFETY_DIST
        return 0.0

    def _compute_best_heading(self, pos_x: float, pos_y: float, cov_trace: float,
                               grid: np.ndarray, gi, clusters: list) -> float:
        """Evaluate N_CANDIDATE_HEADINGS and return the heading with highest potential."""
        headings   = np.linspace(-math.pi, math.pi, N_CANDIDATE_HEADINGS, endpoint=False)
        potentials = np.zeros(N_CANDIDATE_HEADINGS)

        # Normalise covariance for penalty weight
        cov_norm = min(cov_trace / 2.0, 1.0)

        for i, h in enumerate(headings):
            info_gain  = self._ray_info_gain(pos_x, pos_y, h, grid, gi)
            frontier_p = self._frontier_distance_potential(pos_x, pos_y, h, clusters)
            obs_pen    = self._obstacle_penalty(h)
            lc_pot     = self._loop_closure_potential(pos_x, pos_y, h, grid, gi)

            potentials[i] = (
                W_INFO     * info_gain
                + W_FRONTIER * frontier_p
                + W_LC       * lc_pot
                - W_COV      * cov_norm
                - W_OBS      * obs_pen
            )

        # Prefer safe headings
        for i, h in enumerate(headings):
            if not self._check_safety(h):
                potentials[i] = -1e9

        best_idx = int(np.argmax(potentials))
        return float(headings[best_idx])

    def _compute_step(self, pos_x: float, pos_y: float, heading: float):
        """Move MAX_STEP_DIST in the selected heading, with yaw rate limiting."""
        dx = math.cos(heading) * MAX_STEP_DIST
        dy = math.sin(heading) * MAX_STEP_DIST
        nx = float(np.clip(pos_x + dx, WS_MIN_X, WS_MAX_X))
        ny = float(np.clip(pos_y + dy, WS_MIN_Y, WS_MAX_Y))

        yaw_diff   = (heading - self.prev_yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_diff   = float(np.clip(yaw_diff, -MAX_YAW_CHANGE, MAX_YAW_CHANGE))
        target_yaw = (self.prev_yaw + yaw_diff + math.pi) % (2 * math.pi) - math.pi
        self.prev_yaw = target_yaw
        return (nx, ny, target_yaw)

    # ---------------------------------------------------------------- frontier detection (metrics)
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
        print(f"[ITPF] Metrics saved → {filepath} ({len(self.metrics)} rows)")

    # ---------------------------------------------------------------- main loop
    def run(self):
        if not self.arm_and_takeoff():
            return

        print("[ITPF] Starting information-theoretic potential field exploration …")
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

                pos = self.node.get_drone_position()
                if pos is None:
                    time.sleep(STEP_DURATION)
                    self.log_metrics(step, "no_pos")
                    continue

                grid     = self.node.occupancy_grid
                gi       = self.node.grid_info
                cov      = self.node.get_covariance_trace()
                clusters = self._detect_frontiers()

                # Anti-stuck: if no new cells for many steps, perturb direction
                new_cells_now = max(0, self.node.known_cells_count - self.prev_known_cells)
                if new_cells_now == 0:
                    self.stuck_steps += 1
                else:
                    self.stuck_steps = 0

                if self.stuck_steps > MAX_STUCK_STEPS:
                    # Force a large random perturbation
                    self.perturb_angle += math.pi * 0.75
                    heading = self.perturb_angle
                    note    = "perturb"
                else:
                    heading = self._compute_best_heading(
                        pos[0], pos[1], cov, grid, gi, clusters
                    )
                    note = f"itpf_h{math.degrees(heading):.0f}"

                nx, ny, yaw = self._compute_step(pos[0], pos[1], heading)
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
                    ps = f"({pos[0]:.1f},{pos[1]:.1f})"
                    print(
                        f"[Step {step}/{MAX_STEPS}] "
                        f"Cov:{row['coverage']:.1%} "
                        f"Frt:{row['frontier_count']} "
                        f"Pos:{ps} "
                        f"CovTr:{row['cov_trace']:.4f} "
                        f"Stuck:{self.stuck_steps} "
                        f"{elapsed}"
                    )
        except KeyboardInterrupt:
            print("\n[ITPF] Interrupted.")

        elapsed = time.time() - start_time
        final   = self.log_metrics(step if step > 0 else 0, "final")
        print("\n" + "=" * 64)
        print("  POTENTIAL FIELD RESULTS")
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
            PROJECT_ROOT, "logs", f"baseline_potential_field_{timestamp}.csv"
        )
        self.save_metrics(csv_path)

    def shutdown(self):
        try:
            self.node.stop_offboard_stream()
            self.node.land()
            time.sleep(5.0)
            self.node.disarm()
        except Exception as e:
            print(f"[ITPF] Shutdown error: {e}")
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def main():
    explorer = PotentialFieldExplorer()
    try:
        explorer.run()
    finally:
        explorer.shutdown()


if __name__ == "__main__":
    main()
