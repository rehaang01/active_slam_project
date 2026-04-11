#!/usr/bin/env python3
"""
ActiveSLAMEnv: Gymnasium environment for RL-based active SLAM exploration.

Key design principles:
- The SLAM map NEVER resets — it persists and grows across episodes
- Movement is slow and smooth to preserve RTAB-Map visual odometry
- Yaw changes are rate-limited to ~20°/step
- When tracking is lost → retreat to last good position and hover
- When collision detected → retreat to last good position (don't terminate)
- Covariance panic retreat walks backward through trajectory

Fixes applied:
- Non-blocking step() with fixed STEP_DURATION
- Global covariance from /rtabmap/localization_pose
- np.repeat for upscale (not sparse loop)
- Channel-specific padding (0.5 for ch1, 0.0 for ch2/ch3)
- Visited map preserved on grid expansion
- Consistent 7x7 VISIT_RADIUS for marking and checking
- One-shot coverage bonus
- prev_known_cells initialized from current SLAM state
- Yaw = atan2(dy, dx) toward direction of travel (rate-limited)
- Reactive safety: 360° check uses collision dist, directional uses safety dist
- Loop closure reward with cooldown to prevent flooding
- Tracking loss detection with retreat to last good position
"""

import os
import sys
import math
import time
import heapq
import threading
import numpy as np
from collections import deque

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.executors import MultiThreadedExecutor
from px4_msgs.msg import VehicleCommand

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from envs.slam_collector import SLAMDataCollector


# ================================================================
# Configuration Constants
# ================================================================

# --- Map & CNN ---
MAP_SIZE = 64              # CNN input: MAP_SIZE x MAP_SIZE

# --- Flight ---
FLIGHT_ALTITUDE = -1.2     # NED frame: negative = up (1.2m above ground)
                           # Was -1.5 but PX4 can't maintain 1.5m with LiDAR weight
MAX_STEP_DIST = 0.5        # Max displacement per RL action (meters)
MAX_STEPS = 2000           # Was 200 — now long enough to map entire warehouse
                           # At 31% coverage per 200 steps, 2000 steps should reach 90%+
                           # Avoids episode transition which crashes RTAB-Map every time
STEP_DURATION = 2.0        # Seconds to fly before observing

# --- Workspace bounds (warehouse ~16m x 10m) ---
WS_MIN_X = -8.0
WS_MAX_X = 8.0
WS_MIN_Y = -5.0
WS_MAX_Y = 5.0

# --- Z-filter for altitude slicing ---
ENABLE_Z_FILTER = False
Z_SLICE_HALF_HEIGHT = 0.5

# --- Yaw rate limiting ---
MAX_YAW_CHANGE = 0.20      # ~11 degrees per step — very slow turns
                           # Was 0.35 (~20°) — too fast, caused RTAB-Map tracking loss
                           # The OakD-Lite's 70° FOV needs gradual rotations

# --- Altitude management ---
ENABLE_ALTITUDE_MGMT = False
ALT_STEP = 2.0
ALT_MIN = 1.0
ALT_MAX = 5.0
FRONTIER_ZERO_THRESHOLD = 10

# --- Safety thresholds ---
MIN_ALTITUDE = 0.15        # Below this = crashed
MAX_ALTITUDE_NED = -5.0
LIDAR_COLLISION_DIST = 0.35 # Hard collision boundary (was 0.4)
LIDAR_SAFETY_DIST = 0.5    # Was 1.0 — reduced to allow shelf aisle navigation
                           # Warehouse aisles are ~1.5-2.5m wide
                           # At 1.0m the drone couldn't enter any aisle
DEPTH_SAFETY_DIST = 0.5    # Was 1.0 — same reason as above
COV_MAX_FOR_NORM = 2.0     # Covariance normalization ceiling
COV_PANIC_THRESHOLD = 2.0  # Emergency retreat threshold
COV_RECOVERY_THRESHOLD = 0.5

# --- Tracking loss recovery ---
TRACKING_HOVER_STEPS = 5   # Hover at last good position waiting for recovery
MAX_TRACKING_LOST_STEPS = 30  # NEW: Force episode truncation after this many lost steps
MAX_CONSECUTIVE_COLLISIONS = 15  # NEW: Force truncation if stuck in collision loop

# --- Reward weights — REBALANCED to fix loop-closure farming ---
W_INFO_GAIN = 5.0          # Was 2.0 — exploration is now the dominant reward
W_FRONTIER = 3.0           # Was 0.5 — strong pull toward unexplored boundaries
W_COV_PENALTY = -1.5       # Was -2.0 — slightly relaxed so drone dares to explore
W_STEP_COST = -0.1         # Unchanged
W_COLLISION = -10.0        # Unchanged
W_REVISIT = -1.0           # Was -0.3 — heavy penalty for circling visited areas
W_LOOP_CLOSURE = 0.3       # Was 2.0 — drastically cut to stop loop-closure farming
LOOP_CLOSURE_COOLDOWN = 20 # Was 5 — much longer cooldown between loop closure rewards
W_COVERAGE_BONUS = 50.0    # Unchanged
W_SMOOTHNESS = -0.1        # Was -0.2 — less penalty for changing direction toward frontiers
W_FRONTIER_APPROACH = 1.5  # NEW: reward for getting closer to nearest frontier

# --- Altitude correction ---
ALT_CORRECTION_THRESHOLD = 0.20  # Correct altitude if > 0.20m off target
ALT_CORRECTION_TIMEOUT = 3.0     # Max time to spend correcting altitude per step

# --- Visited/revisit radius (consistent 7x7 grid) ---
VISIT_RADIUS = 3

# --- Number of scalar features ---
N_SCALARS = 8


class ActiveSLAMEnv(gym.Env):
    """Gymnasium environment for active SLAM with UAV.

    Key behaviors:
    - Map persists across episodes (never resets)
    - Tracking loss → retreat to last good position, hover
    - Collision → retreat to last good position
    - Slow, smooth movement to preserve visual odometry
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "map_tensor": spaces.Box(
                low=0.0, high=1.0,
                shape=(3, MAP_SIZE, MAP_SIZE),
                dtype=np.float32,
            ),
            "scalars": spaces.Box(
                low=0.0, high=1.0,
                shape=(N_SCALARS,),
                dtype=np.float32,
            ),
        })

        # ROS2 init
        if not rclpy.ok():
            rclpy.init()

        self.ros_node = SLAMDataCollector()
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.ros_node)
        self.spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        # Episode state
        self.step_count = 0
        self.episode_count = 0
        self.prev_known_cells = 0
        self.prev_cov_trace = 0.0
        self.prev_loop_closure_id = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_yaw = 0.0
        self.trajectory = []
        self.visited_map = None
        self.drone_armed = False
        self.coverage_bonus_given = False
        self.last_loop_closure_step = -LOOP_CLOSURE_COOLDOWN

        # Altitude management state
        self.current_flight_altitude = abs(FLIGHT_ALTITUDE)
        self.zero_frontier_steps = 0

        # Covariance panic retreat state
        self.panic_mode = False
        self.retreat_index = 0

        # Tracking loss state
        self.tracking_lost_steps = 0
        self.last_good_position = None

        # Consecutive collision counter (NEW — truncate if stuck)
        self.consecutive_collision_steps = 0

        # Frontier approach tracking (NEW — reward getting closer to frontiers)
        self.prev_nearest_frontier_dist = float('inf')

        # Stale data detection (NEW — detect silent RTAB-Map failure)
        self._stale_cov_trace = None
        self._stale_data_steps = 0
        STALE_DATA_THRESHOLD = 8  # If data frozen for 8 steps, declare silent failure

        # Frontier detection cache
        self.frontier_clusters = []
        self.nearest_frontier_dist = float('inf')
        self.nearest_frontier_dir = 0.0

        # Wait for initial data
        self._wait_for_data()

        if ENABLE_Z_FILTER:
            self._apply_z_filter(self.current_flight_altitude)

    def _wait_for_data(self, timeout=30.0):
        """Wait until initial sensor data arrives from SLAM and PX4."""
        start = time.time()
        while time.time() - start < timeout:
            pos_ok = self.ros_node.local_position is not None
            map_ok = self.ros_node.occupancy_grid is not None
            if pos_ok and map_ok:
                print("[ActiveSLAMEnv] Sensor data received (position + map).")
                return True
            time.sleep(0.5)
            status = f"pos={'Y' if pos_ok else 'N'} map={'Y' if map_ok else 'N'}"
            print(f"[ActiveSLAMEnv] Waiting for data... ({status})")
        print("[ActiveSLAMEnv] WARNING: Timeout waiting for sensor data.")
        return False

    # ================================================================
    # Z-Filter Management
    # ================================================================
    def _apply_z_filter(self, altitude_m):
        z_min = altitude_m - Z_SLICE_HALF_HEIGHT
        z_max = altitude_m + Z_SLICE_HALF_HEIGHT
        success = self.ros_node.set_z_filter(z_min, z_max)
        if success:
            print(f"[ActiveSLAMEnv] Z-filter: [{z_min:.1f}m, {z_max:.1f}m] at alt={altitude_m:.1f}m")
        return success

    # ================================================================
    # Observation Building
    # ================================================================
    def _get_observation(self):
        grid = self.ros_node.occupancy_grid
        if grid is None:
            return self._empty_observation()

        h, w = grid.shape

        # Channel 1: Occupancy (obstacles=1, free=0, unknown=0.5)
        ch1 = np.full((h, w), 0.5, dtype=np.float32)
        ch1[grid == 0] = 0.0
        ch1[(grid > 0) & (grid <= 100)] = 1.0

        # Channel 2: Visited regions
        self._update_visited_map(h, w)
        ch2 = self.visited_map.copy()

        # Channel 3: Drone position + recent trajectory
        ch3 = np.zeros((h, w), dtype=np.float32)
        pos = self.ros_node.get_drone_position()
        if pos is not None and self.ros_node.grid_info is not None:
            gi = self.ros_node.grid_info
            gx = int((pos[0] - gi.origin.position.x) / gi.resolution)
            gy = int((pos[1] - gi.origin.position.y) / gi.resolution)

            if 0 <= gx < w and 0 <= gy < h:
                ch3[gy, gx] = 1.0

            n_trail = min(len(self.trajectory), 20)
            for i, (tx, ty) in enumerate(self.trajectory[-n_trail:]):
                tgx = int((tx - gi.origin.position.x) / gi.resolution)
                tgy = int((ty - gi.origin.position.y) / gi.resolution)
                if 0 <= tgx < w and 0 <= tgy < h:
                    decay = (i + 1) / n_trail
                    ch3[tgy, tgx] = max(ch3[tgy, tgx], decay * 0.8)

        ch1_r = self._resize_channel(ch1, MAP_SIZE, pad_value=0.5)
        ch2_r = self._resize_channel(ch2, MAP_SIZE, pad_value=0.0)
        ch3_r = self._resize_channel(ch3, MAP_SIZE, pad_value=0.0)

        map_tensor = np.stack([ch1_r, ch2_r, ch3_r], axis=0)

        self._detect_frontiers(grid)

        # Scalar features (8 values, all normalized to ~[0, 1])
        cov_norm = self.ros_node.get_covariance_trace_normalized(COV_MAX_FOR_NORM)
        n_frontier_cells = sum(c[2] for c in self.frontier_clusters)
        frontier_count_norm = min(n_frontier_cells / 100.0, 1.0)
        coverage = self.ros_node.known_cells_count / max(self.ros_node.total_cells, 1)
        frontier_dist_norm = min(self.nearest_frontier_dist / 10.0, 1.0)
        frontier_dir_norm = (self.nearest_frontier_dir + np.pi) / (2 * np.pi)
        alt = self.ros_node.get_altitude()
        altitude_norm = np.clip((alt if alt is not None else ALT_MIN) / ALT_MAX, 0, 1)

        if pos is not None:
            dist_from_center = math.sqrt(pos[0]**2 + pos[1]**2)
            pos_norm = min(dist_from_center / 10.0, 1.0)
        else:
            pos_norm = 0.5

        step_frac = self.step_count / MAX_STEPS

        scalars = np.array([
            cov_norm, frontier_count_norm, coverage,
            frontier_dist_norm, frontier_dir_norm,
            altitude_norm, pos_norm, step_frac,
        ], dtype=np.float32)

        return {"map_tensor": map_tensor, "scalars": scalars}

    def _update_visited_map(self, h, w):
        if self.visited_map is None:
            self.visited_map = np.zeros((h, w), dtype=np.float32)
        elif self.visited_map.shape != (h, w):
            new_vm = np.zeros((h, w), dtype=np.float32)
            oh, ow = self.visited_map.shape
            copy_h, copy_w = min(oh, h), min(ow, w)
            new_vm[:copy_h, :copy_w] = self.visited_map[:copy_h, :copy_w]
            self.visited_map = new_vm

        pos = self.ros_node.get_drone_position()
        if pos is not None and self.ros_node.grid_info is not None:
            gi = self.ros_node.grid_info
            gx = int((pos[0] - gi.origin.position.x) / gi.resolution)
            gy = int((pos[1] - gi.origin.position.y) / gi.resolution)
            for dx in range(-VISIT_RADIUS, VISIT_RADIUS + 1):
                for dy in range(-VISIT_RADIUS, VISIT_RADIUS + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        self.visited_map[ny, nx] = 1.0

    def _resize_channel(self, channel, target_size, pad_value=0.5):
        h, w = channel.shape
        if h == target_size and w == target_size:
            return channel.astype(np.float32)

        max_dim = max(h, w)
        padded = np.full((max_dim, max_dim), pad_value, dtype=np.float32)
        padded[:h, :w] = channel

        if max_dim >= target_size:
            block = max_dim // target_size
            if block < 1:
                block = 1
            trimmed_size = block * target_size
            trimmed = padded[:trimmed_size, :trimmed_size]
            resized = trimmed.reshape(
                target_size, block, target_size, block
            ).mean(axis=(1, 3))
        else:
            factor = max(1, math.ceil(target_size / max_dim))
            upscaled = np.repeat(np.repeat(padded, factor, axis=0), factor, axis=1)
            resized = upscaled[:target_size, :target_size]

        return resized.astype(np.float32)

    def _empty_observation(self):
        return {
            "map_tensor": np.full((3, MAP_SIZE, MAP_SIZE), 0.0, dtype=np.float32),
            "scalars": np.zeros(N_SCALARS, dtype=np.float32),
        }

    # ================================================================
    # Frontier Detection (Connected Components on Z-Slice)
    # ================================================================
    def _detect_frontiers(self, grid):
        h, w = grid.shape
        free_mask = (grid == 0)
        unknown_mask = (grid == -1)

        unknown_adjacent = np.zeros_like(unknown_mask)
        unknown_adjacent[1:, :] |= unknown_mask[:-1, :]
        unknown_adjacent[:-1, :] |= unknown_mask[1:, :]
        unknown_adjacent[:, 1:] |= unknown_mask[:, :-1]
        unknown_adjacent[:, :-1] |= unknown_mask[:, 1:]

        frontier_mask = free_mask & unknown_adjacent

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

                    if len(cells) >= 2:
                        cx_mean = np.mean([c[0] for c in cells])
                        cy_mean = np.mean([c[1] for c in cells])
                        clusters.append((cx_mean, cy_mean, len(cells)))

        self.frontier_clusters = []
        if self.ros_node.grid_info is not None:
            gi = self.ros_node.grid_info
            for cx_grid, cy_grid, size in clusters:
                wx = cx_grid * gi.resolution + gi.origin.position.x
                wy = cy_grid * gi.resolution + gi.origin.position.y
                self.frontier_clusters.append((wx, wy, size))

        pos = self.ros_node.get_drone_position()
        self.nearest_frontier_dist = float('inf')
        self.nearest_frontier_dir = 0.0

        if pos is not None and len(self.frontier_clusters) > 0:
            min_dist = float('inf')
            min_dir = 0.0
            for fx, fy, fsize in self.frontier_clusters:
                dist = math.sqrt((fx - pos[0])**2 + (fy - pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    min_dir = math.atan2(fy - pos[1], fx - pos[0])
            self.nearest_frontier_dist = min_dist
            self.nearest_frontier_dir = min_dir

    # ================================================================
    # Safety Checks (LiDAR + Depth Camera)
    # ================================================================
    def _check_lidar_collision(self):
        return self.ros_node.get_min_lidar_range() < LIDAR_COLLISION_DIST

    def _check_reactive_safety(self, target_yaw=None):
        """Check if the drone should hover due to proximity to obstacles.
        360° global check uses COLLISION dist (tight).
        Directional check uses SAFETY dist (wider)."""
        if self.ros_node.get_min_lidar_range() < LIDAR_COLLISION_DIST:
            return True

        if target_yaw is not None:
            range_ahead = self.ros_node.get_lidar_range_in_direction(target_yaw)
            if range_ahead < LIDAR_SAFETY_DIST:
                return True

        if self.ros_node.get_depth_min_distance() < DEPTH_SAFETY_DIST:
            return True

        return False

    # ================================================================
    # Path Validation: 2D Raytrace + A*
    # ================================================================
    def _world_to_grid(self, wx, wy):
        gi = self.ros_node.grid_info
        if gi is None:
            return None
        gx = int((wx - gi.origin.position.x) / gi.resolution)
        gy = int((wy - gi.origin.position.y) / gi.resolution)
        return gx, gy

    def _grid_to_world(self, gx, gy):
        gi = self.ros_node.grid_info
        if gi is None:
            return None
        wx = gx * gi.resolution + gi.origin.position.x + gi.resolution / 2
        wy = gy * gi.resolution + gi.origin.position.y + gi.resolution / 2
        return wx, wy

    def _is_cell_free(self, gx, gy):
        grid = self.ros_node.occupancy_grid
        if grid is None:
            return False
        h, w = grid.shape
        if not (0 <= gx < w and 0 <= gy < h):
            return False
        return grid[gy, gx] == 0

    def _raytrace_2d(self, x0, y0, x1, y1):
        grid = self.ros_node.occupancy_grid
        if grid is None:
            return True

        h, w = grid.shape
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cx, cy = x0, y0
        while True:
            if 0 <= cx < w and 0 <= cy < h:
                if grid[cy, cx] > 0:
                    return False
            else:
                return False

            if cx == x1 and cy == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

        return True

    def _astar_path(self, start_gx, start_gy, goal_gx, goal_gy, max_iterations=2000):
        grid = self.ros_node.occupancy_grid
        if grid is None:
            return None
        h, w = grid.shape

        if not (0 <= start_gx < w and 0 <= start_gy < h):
            return None
        if not (0 <= goal_gx < w and 0 <= goal_gy < h):
            return None

        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        start = (start_gx, start_gy)
        goal = (goal_gx, goal_gy)

        open_set = [(0 + heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            f, g, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in closed:
                continue
            closed.add(current)

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if grid[ny, nx] > 0:
                    continue
                if grid[ny, nx] < 0:
                    move_cost = 2.0
                else:
                    move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0

                tentative_g = g + move_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None

    def _validate_and_plan_path(self, start_x, start_y, target_x, target_y):
        start_grid = self._world_to_grid(start_x, start_y)
        target_grid = self._world_to_grid(target_x, target_y)

        if start_grid is None or target_grid is None:
            return [(target_x, target_y)]

        sgx, sgy = start_grid
        tgx, tgy = target_grid

        if self._raytrace_2d(sgx, sgy, tgx, tgy):
            return [(target_x, target_y)]

        path = self._astar_path(sgx, sgy, tgx, tgy)
        if path is None or len(path) == 0:
            return [(target_x, target_y)]

        step = max(1, len(path) // 5)
        world_waypoints = []
        for i in range(0, len(path), step):
            wpt = self._grid_to_world(path[i][0], path[i][1])
            if wpt is not None:
                world_waypoints.append(wpt)

        world_waypoints.append((target_x, target_y))
        return world_waypoints

    # ================================================================
    # Action Optimization Unit (AOU)
    # ================================================================
    def _snap_to_free_point(self, target_x, target_y, search_radius=2.0):
        grid = self.ros_node.occupancy_grid
        gi = self.ros_node.grid_info
        if grid is None or gi is None:
            return target_x, target_y

        tg = self._world_to_grid(target_x, target_y)
        if tg is None:
            return target_x, target_y

        tgx, tgy = tg
        h, w = grid.shape

        if 0 <= tgx < w and 0 <= tgy < h and grid[tgy, tgx] == 0:
            return target_x, target_y

        search_cells = int(search_radius / gi.resolution)
        best_dist = float('inf')
        best_gx, best_gy = tgx, tgy

        for dy in range(-search_cells, search_cells + 1):
            for dx in range(-search_cells, search_cells + 1):
                nx, ny = tgx + dx, tgy + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_gx, best_gy = nx, ny

        result = self._grid_to_world(best_gx, best_gy)
        if result is None:
            return target_x, target_y
        return result

    # ================================================================
    # Altitude Management
    # ================================================================
    def _check_altitude_bump(self):
        if not ENABLE_ALTITUDE_MGMT:
            return False

        n_frontier_cells = sum(c[2] for c in self.frontier_clusters)

        if n_frontier_cells == 0:
            self.zero_frontier_steps += 1
        else:
            self.zero_frontier_steps = 0

        if self.zero_frontier_steps >= FRONTIER_ZERO_THRESHOLD:
            new_alt = self.current_flight_altitude + ALT_STEP
            if new_alt <= ALT_MAX:
                self.current_flight_altitude = new_alt
                if ENABLE_Z_FILTER:
                    self._apply_z_filter(new_alt)
                self.zero_frontier_steps = 0
                print(f"[ActiveSLAMEnv] Altitude bumped to {new_alt:.1f}m (0 frontiers)")
                return True

        return False

    def _correct_altitude_if_needed(self):
        """Check actual altitude and correct before lateral movement.
        Loops until altitude is within tolerance or timeout expires.
        Fixes Bug 2: drone drifting/landing due to LiDAR weight + lateral accel."""
        alt = self.ros_node.get_altitude()
        if alt is None:
            return

        target_alt = self.current_flight_altitude  # 1.5m
        deviation = abs(alt - target_alt)

        if deviation <= ALT_CORRECTION_THRESHOLD:
            return  # Altitude is fine

        # Hold current XY position and enforce correct Z
        pos = self.ros_node.get_drone_position()
        if pos is None:
            return

        corrected_z_ned = -target_alt
        start_time = time.time()
        initial_alt = alt

        # Keep sending altitude-correction setpoint until within tolerance or timeout
        while time.time() - start_time < ALT_CORRECTION_TIMEOUT:
            self.ros_node.update_setpoint(
                pos[0], pos[1], corrected_z_ned, yaw=self.prev_yaw)
            time.sleep(0.2)

            alt = self.ros_node.get_altitude()
            if alt is not None and abs(alt - target_alt) <= ALT_CORRECTION_THRESHOLD:
                break

        final_alt = self.ros_node.get_altitude()
        elapsed = time.time() - start_time
        if final_alt is not None and self.step_count % 10 == 0:
            print(f"[ActiveSLAMEnv] Altitude corrected: {initial_alt:.2f}m → {final_alt:.2f}m "
                  f"(target: {target_alt:.1f}m, took {elapsed:.1f}s)")

    # ================================================================
    # Covariance Panic Retreat
    # ================================================================
    def _check_covariance_retreat(self):
        cov = self.ros_node.get_covariance_trace()

        if not self.panic_mode:
            if cov > COV_PANIC_THRESHOLD and len(self.trajectory) > 2:
                self.panic_mode = True
                self.retreat_index = max(0, len(self.trajectory) - 3)
                print(f"[ActiveSLAMEnv] PANIC: cov={cov:.4f} > {COV_PANIC_THRESHOLD}, retreating")
        else:
            if cov < COV_RECOVERY_THRESHOLD:
                self.panic_mode = False
                self.retreat_index = 0
                print(f"[ActiveSLAMEnv] RECOVERED: cov={cov:.4f} < {COV_RECOVERY_THRESHOLD}")
                return False, None

        if self.panic_mode and self.retreat_index >= 0 and len(self.trajectory) > 0:
            rx, ry = self.trajectory[self.retreat_index]
            self.retreat_index = max(0, self.retreat_index - 1)
            z_ned = -self.current_flight_altitude
            return True, (rx, ry, z_ned)

        return False, None

    # ================================================================
    # Revisit Ratio
    # ================================================================
    def _get_revisit_ratio(self):
        if self.visited_map is None:
            return 0.0

        pos = self.ros_node.get_drone_position()
        if pos is None or self.ros_node.grid_info is None:
            return 0.0

        gi = self.ros_node.grid_info
        gx = int((pos[0] - gi.origin.position.x) / gi.resolution)
        gy = int((pos[1] - gi.origin.position.y) / gi.resolution)
        h, w = self.visited_map.shape

        total = 0
        visited = 0
        for dx in range(-VISIT_RADIUS, VISIT_RADIUS + 1):
            for dy in range(-VISIT_RADIUS, VISIT_RADIUS + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    total += 1
                    if self.visited_map[ny, nx] > 0:
                        visited += 1

        return visited / max(total, 1)

    # ================================================================
    # Drone Lifecycle: Arm, Takeoff, Land
    # ================================================================
    def _arm_and_takeoff(self, target_altitude_ned=None):
        """Robust arm and takeoff with multiple retry attempts.

        Key improvements over original:
        - 3.0s pre-arm heartbeats (was 2.5s — PX4 needs time after crash)
        - Multiple arm attempts with delays
        - Altitude hold verification after reaching target
        - Post-takeoff stabilization hover
        """
        if target_altitude_ned is None:
            target_altitude_ned = -self.current_flight_altitude

        target_alt_positive = -target_altitude_ned
        print(f"[ActiveSLAMEnv] Arming and taking off to {target_alt_positive:.1f}m...")

        # Pre-arm: send offboard heartbeats for 3s so PX4 accepts offboard mode
        self.ros_node.start_offboard_stream(0.0, 0.0, target_altitude_ned, yaw=0.0)
        time.sleep(3.0)

        # Engage offboard + arm with retry
        for attempt in range(3):
            self.ros_node.engage_offboard()
            time.sleep(0.3)
            self.ros_node.arm()
            time.sleep(0.5)

            arm_timeout = 3.0
            arm_start = time.time()
            while time.time() - arm_start < arm_timeout:
                if self.ros_node.is_armed():
                    self.drone_armed = True
                    print(f"[ActiveSLAMEnv] Drone ARMED successfully (attempt {attempt + 1}).")
                    break
                time.sleep(0.1)

            if self.drone_armed:
                break
            print(f"[ActiveSLAMEnv] Arm attempt {attempt + 1} failed, retrying...")
            time.sleep(1.0)

        if not self.drone_armed:
            print("[ActiveSLAMEnv] WARNING: Arming failed after 3 attempts.")
            return False

        # Wait for altitude — 30s timeout
        alt_timeout = 30.0
        alt_start = time.time()
        while time.time() - alt_start < alt_timeout:
            alt = self.ros_node.get_altitude()
            if alt is not None and abs(alt - target_alt_positive) < 0.2:
                print(f"[ActiveSLAMEnv] Reached altitude: {alt:.2f}m "
                      f"(target: {target_alt_positive:.1f}m)")
                break
            # Keep sending the setpoint
            self.ros_node.update_setpoint(0.0, 0.0, target_altitude_ned, yaw=0.0)
            time.sleep(0.1)
        else:
            alt = self.ros_node.get_altitude()
            if alt is not None and alt > 0.5:
                print(f"[ActiveSLAMEnv] Altitude close enough: {alt:.2f}m "
                      f"(target: {target_alt_positive:.1f}m)")
            else:
                print(f"[ActiveSLAMEnv] WARNING: Altitude timeout. Current: {alt}m")
                return False

        # Post-takeoff stabilization — hold for 3s before any movement
        # This is CRITICAL: gives PX4 time to stabilize attitude and
        # gives RTAB-Map time to start processing frames from the new position
        print("[ActiveSLAMEnv] Post-takeoff stabilization (3s)...")
        for _ in range(30):
            self.ros_node.update_setpoint(0.0, 0.0, target_altitude_ned, yaw=0.0)
            time.sleep(0.1)

        final_alt = self.ros_node.get_altitude()
        print(f"[ActiveSLAMEnv] Stabilized at altitude: {final_alt:.2f}m")
        return True

    def close(self):
        print("[ActiveSLAMEnv] Closing environment...")

        try:
            self.ros_node.stop_offboard_stream()
            self.ros_node.land()
            print("[ActiveSLAMEnv] Land command sent.")

            land_timeout = 15.0
            land_start = time.time()
            while time.time() - land_start < land_timeout:
                alt = self.ros_node.get_altitude()
                if alt is not None and alt < 0.2:
                    print("[ActiveSLAMEnv] Drone landed.")
                    break
                time.sleep(0.2)

            self.ros_node.disarm()
            time.sleep(0.5)
            self.drone_armed = False
            print("[ActiveSLAMEnv] Drone disarmed.")

        except Exception as e:
            print(f"[ActiveSLAMEnv] Error during close: {e}")

        try:
            self.executor.shutdown()
            self.ros_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            print("[ActiveSLAMEnv] ROS2 shutdown complete.")
        except Exception as e:
            print(f"[ActiveSLAMEnv] ROS2 shutdown error (non-fatal): {e}")

    # ================================================================
    # Episode Reset — Map NEVER resets, only episode state
    # ================================================================
    def reset(self, seed=None, options=None):
        """Reset for a new episode. The SLAM map persists and grows.

        CRITICAL DESIGN DECISION: Episode transitions MUST be minimal.
        RTAB-Map corrupts during aggressive maneuvers between episodes
        (return-to-origin, orientation pushes, re-arming). Once corrupted,
        PX4's position estimate drifts permanently — unrecoverable.

        Strategy:
        - Episode 1: Clean arm and takeoff (always works)
        - Episode 2+: If drone is flying → just reset RL state, keep flying
                       If drone crashed → attempt simple re-arm
        - NEVER return to origin between episodes
        - NEVER do aggressive orientation maneuvers between episodes
        """
        super().reset(seed=seed)
        self.episode_count += 1
        print(f"\n[ActiveSLAMEnv] === RESET (Episode {self.episode_count}) ===")

        alt = self.ros_node.get_altitude()
        pos = self.ros_node.get_drone_position()

        if self.episode_count == 1:
            # First episode — clean takeoff (this always works)
            print("[ActiveSLAMEnv] First episode — arming and taking off...")
            self._arm_and_takeoff()

        elif alt is not None and alt < MIN_ALTITUDE:
            # Drone crashed — attempt re-arm
            print(f"[ActiveSLAMEnv] Drone on ground (alt={alt:.2f}m). Attempting re-arm...")
            self.drone_armed = False
            time.sleep(2.0)
            success = self._arm_and_takeoff()
            if not success:
                print("[ActiveSLAMEnv] Re-arm failed. Continuing with whatever state we have.")

        elif self.drone_armed and pos is not None:
            # Drone is flying — DO NOT return to origin, DO NOT move
            # Just hold current position for a moment to stabilize
            print(f"[ActiveSLAMEnv] Drone flying at alt={alt:.2f}m. "
                  f"Holding position for new episode...")
            origin_z = -self.current_flight_altitude
            for _ in range(20):  # 2 seconds of stable hover
                self.ros_node.update_setpoint(pos[0], pos[1], origin_z, yaw=self.prev_yaw)
                time.sleep(0.1)

        else:
            print("[ActiveSLAMEnv] Unknown state — arming and taking off...")
            self._arm_and_takeoff()

        # Phase 2: Reset episode state variables (NOT the SLAM map)
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        # Read the drone's ACTUAL heading — not 0.0.
        # Hardcoding 0.0 caused a sudden yaw snap on episode 2+ when the
        # stabilization hover (below) sent yaw=0.0 while the drone was
        # at a completely different heading.  The snap breaks RTAB-Map
        # feature matching → tracking loss at the very start of the episode.
        self.prev_yaw = self.ros_node.get_drone_yaw()
        self.trajectory = []
        self.coverage_bonus_given = False
        self.prev_loop_closure_id = self.ros_node.loop_closure_id
        self.last_loop_closure_step = -LOOP_CLOSURE_COOLDOWN

        self.current_flight_altitude = abs(FLIGHT_ALTITUDE)
        self.zero_frontier_steps = 0

        self.panic_mode = False
        self.retreat_index = 0

        self.tracking_lost_steps = 0
        self.last_good_position = None

        # NEW state resets
        self.consecutive_collision_steps = 0
        self.prev_nearest_frontier_dist = float('inf')
        self._stale_cov_trace = None
        self._stale_data_steps = 0

        self.visited_map = None

        self.frontier_clusters = []
        self.nearest_frontier_dist = float('inf')
        self.nearest_frontier_dir = 0.0

        if ENABLE_Z_FILTER:
            self._apply_z_filter(self.current_flight_altitude)

        # Brief stabilization — actively hold position
        origin_z = -self.current_flight_altitude
        pos = self.ros_node.get_drone_position()
        if pos is not None and self.drone_armed:
            for _ in range(10):  # 1 second
                self.ros_node.update_setpoint(pos[0], pos[1], origin_z, yaw=self.prev_yaw)
                time.sleep(0.1)

        # Phase 3: Build initial observation and initialize baselines
        obs = self._get_observation()

        self.prev_known_cells = self.ros_node.known_cells_count
        self.prev_cov_trace = self.ros_node.get_covariance_trace()

        pos = self.ros_node.get_drone_position()
        if pos is not None:
            self.trajectory.append((pos[0], pos[1]))
            self.last_good_position = (pos[0], pos[1])

        # NEW: Initialize frontier approach tracking
        self.prev_nearest_frontier_dist = self.nearest_frontier_dist

        info = {
            "episode": self.episode_count,
            "initial_coverage": self.ros_node.known_cells_count / max(self.ros_node.total_cells, 1),
            "initial_cov_trace": self.prev_cov_trace,
            "altitude": self.current_flight_altitude,
        }

        print(f"[ActiveSLAMEnv] Episode {self.episode_count} ready. "
              f"Known cells: {self.prev_known_cells}, "
              f"Cov: {self.prev_cov_trace:.4f}")

        return obs, info

    # ================================================================
    # Reward Computation (8 Components)
    # ================================================================
    def _compute_reward(self, action, collision_occurred=False):
        reward = 0.0
        info = {}

        # Component 1: Information Gain — PRIMARY exploration driver
        current_known = self.ros_node.known_cells_count
        new_cells = max(0, current_known - self.prev_known_cells)
        info_gain = W_INFO_GAIN * math.log1p(new_cells)
        reward += info_gain
        info["r_info_gain"] = info_gain
        info["new_cells"] = new_cells
        self.prev_known_cells = current_known

        # Component 2: Frontier Attraction — score based on size AND proximity
        frontier_reward = 0.0
        if len(self.frontier_clusters) > 0 and self.nearest_frontier_dist < float('inf'):
            best_score = 0.0
            for fx, fy, fsize in self.frontier_clusters:
                pos = self.ros_node.get_drone_position()
                if pos is not None:
                    dist = math.sqrt((fx - pos[0])**2 + (fy - pos[1])**2)
                    # Larger clusters at closer range score higher
                    score = fsize / (dist + 0.5)  # Was dist+1.0, now more sensitive to proximity
                    best_score = max(best_score, score)
            frontier_reward = W_FRONTIER * min(best_score / 5.0, 1.0)  # Was /10, now reaches max easier
        reward += frontier_reward
        info["r_frontier"] = frontier_reward

        # Component 2b: Frontier Approach — NEW — reward for getting CLOSER to frontiers
        frontier_approach_reward = 0.0
        current_frontier_dist = self.nearest_frontier_dist
        if (current_frontier_dist < float('inf')
                and self.prev_nearest_frontier_dist < float('inf')):
            approach_delta = self.prev_nearest_frontier_dist - current_frontier_dist
            if approach_delta > 0:
                # Got closer to frontier — reward
                frontier_approach_reward = W_FRONTIER_APPROACH * min(approach_delta, 1.0)
            else:
                # Moved away from frontier — small penalty
                frontier_approach_reward = -0.3 * min(abs(approach_delta), 1.0)
        self.prev_nearest_frontier_dist = current_frontier_dist
        reward += frontier_approach_reward
        info["r_frontier_approach"] = frontier_approach_reward

        # Component 3: Covariance Growth Penalty
        current_cov = self.ros_node.get_covariance_trace()
        cov_growth = current_cov - self.prev_cov_trace
        cov_penalty = W_COV_PENALTY * max(cov_growth, 0.0)
        reward += cov_penalty
        info["r_cov_penalty"] = cov_penalty
        info["cov_trace"] = current_cov
        info["cov_growth"] = cov_growth
        self.prev_cov_trace = current_cov

        # Component 4: Step Cost
        reward += W_STEP_COST
        info["r_step_cost"] = W_STEP_COST

        # Component 5: Loop Closure Bonus (DRASTICALLY reduced + long cooldown)
        loop_closure_reward = 0.0
        current_lc_id = self.ros_node.loop_closure_id
        steps_since_last_lc = self.step_count - self.last_loop_closure_step
        if (current_lc_id != 0
                and current_lc_id != self.prev_loop_closure_id
                and steps_since_last_lc >= LOOP_CLOSURE_COOLDOWN):
            loop_closure_reward = W_LOOP_CLOSURE
            self.last_loop_closure_step = self.step_count
            # Only print occasionally to reduce log spam
            if self.step_count % 20 == 0:
                print(f"[Reward] Loop closure! ID={current_lc_id}, +{W_LOOP_CLOSURE}")
        self.prev_loop_closure_id = current_lc_id
        reward += loop_closure_reward
        info["r_loop_closure"] = loop_closure_reward

        # Component 6: Collision Penalty
        collision_penalty = 0.0
        if collision_occurred:
            collision_penalty = W_COLLISION
        reward += collision_penalty
        info["r_collision"] = collision_penalty

        # Component 7: Revisit Penalty — STRONGER to punish circling
        revisit_ratio = self._get_revisit_ratio()
        revisit_penalty = W_REVISIT * revisit_ratio
        reward += revisit_penalty
        info["r_revisit"] = revisit_penalty
        info["revisit_ratio"] = revisit_ratio

        # Component 8: Smoothness Penalty
        action_change = np.linalg.norm(action - self.prev_action)
        smoothness_penalty = W_SMOOTHNESS * action_change
        reward += smoothness_penalty
        info["r_smoothness"] = smoothness_penalty

        # Coverage Bonus (one-shot at 90%)
        coverage = current_known / max(self.ros_node.total_cells, 1)
        if coverage > 0.9 and not self.coverage_bonus_given:
            reward += W_COVERAGE_BONUS
            self.coverage_bonus_given = True
            info["r_coverage_bonus"] = W_COVERAGE_BONUS
            print(f"[Reward] Coverage bonus triggered at {coverage:.1%}!")
        else:
            info["r_coverage_bonus"] = 0.0

        info["coverage"] = coverage
        info["total_reward"] = reward

        return reward, info

    # ================================================================
    # Common Info Builder — guarantees eval_rl.py gets every field
    # ================================================================
    def _build_common_info(self):
        """Build the info dict fields that eval_rl.py and plot_comparison.py
        need in EVERY step, regardless of which code path was taken.

        This is called from every return path in step() so that the CSV
        output schema is always complete.  Path-specific fields (e.g.
        reward components from _compute_reward) are merged on top
        afterward — they override any duplicate keys set here.
        """
        pos = self.ros_node.get_drone_position()
        alt = self.ros_node.get_altitude()
        coverage = self.ros_node.known_cells_count / max(self.ros_node.total_cells, 1)
        return {
            # ---- fields eval_rl.py reads directly ----
            "known_cells":           self.ros_node.known_cells_count,
            "total_cells":           self.ros_node.total_cells,
            "coverage":              coverage,
            "cov_trace":             self.ros_node.get_covariance_trace(),
            "frontier_count":        len(self.frontier_clusters),
            "nearest_frontier_dist": self.nearest_frontier_dist,
            "loop_closure_id":       self.ros_node.loop_closure_id,
            "pos_x":                 float(pos[0]) if pos is not None else 0.0,
            "pos_y":                 float(pos[1]) if pos is not None else 0.0,
            "altitude":              float(alt) if alt is not None else 0.0,
            "tracking_lost":         self.ros_node.is_tracking_lost(),
            "new_cells":             0,
            # ---- fields used by logging / debug ----
            "panic_mode":            self.panic_mode,
            "flight_altitude":       self.current_flight_altitude,
            "step":                  self.step_count,
        }

    # ================================================================
    # RL Step — with tracking loss protection
    # ================================================================
    def step(self, action):
        """Execute one RL step with tracking loss protection.

        Key safety behaviors (REVISED):
        - Altitude correction BEFORE lateral movement (fixes drift bug)
        - If RTAB-Map tracking lost → retreat to ORIGIN (0,0) not last_good_position
        - If tracking lost > 30 steps → force episode truncation
        - If collision → retreat + counter; > 15 consecutive → truncation
        - Yaw rate-limited to ~20°/step
        - Movement slow (0.5m/step, 2s duration)
        """
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        terminated = False
        truncated = False
        collision_occurred = False

        pos = self.ros_node.get_drone_position()
        if pos is None:
            obs = self._empty_observation()
            info = self._build_common_info()
            info["error"] = "no_position_data"
            return obs, W_COLLISION, True, False, info

        current_x, current_y, current_z = pos[0], pos[1], pos[2]
        flight_z_ned = -self.current_flight_altitude

        # ----------------------------------------------------------
        # 0. NEW: Altitude correction — fix drift before lateral move
        # ----------------------------------------------------------
        self._correct_altitude_if_needed()

        # ----------------------------------------------------------
        # 1. Tracking Loss Detection — HOVER IN PLACE and wait
        #    Previous approach flew to origin (0,0), but this caused
        #    crashes: the aggressive lateral move destabilizes PX4's
        #    altitude controller (LiDAR weight), drone descends and
        #    hits ground.  The baselines survive tracking loss by
        #    hovering in place — we do the same.
        # ----------------------------------------------------------
        if self.ros_node.is_tracking_lost():
            self.tracking_lost_steps += 1

            # Force truncation if stuck in tracking-lost state too long
            if self.tracking_lost_steps >= MAX_TRACKING_LOST_STEPS:
                obs = self._get_observation()
                info = self._build_common_info()
                info["tracking_lost"] = True
                info["tracking_lost_steps"] = self.tracking_lost_steps
                info["total_reward"] = W_STEP_COST * 2
                print(f"[ActiveSLAMEnv] Tracking lost for {self.tracking_lost_steps} steps — truncating episode")
                return obs, float(W_STEP_COST * 2), False, True, info

            # HOVER IN PLACE — hold current position, do NOT move
            # This is the safe strategy proven by baseline runs.
            # Moving during tracking loss destabilizes PX4 altitude control.
            self.ros_node.update_setpoint(
                current_x, current_y, flight_z_ned, yaw=self.prev_yaw)

            if self.tracking_lost_steps == 1:
                print(f"[ActiveSLAMEnv] Tracking lost — hovering in place at "
                      f"({current_x:.1f}, {current_y:.1f})")

            time.sleep(STEP_DURATION)

            new_pos = self.ros_node.get_drone_position()
            if new_pos is not None:
                self.trajectory.append((new_pos[0], new_pos[1]))

            obs = self._get_observation()
            reward = W_STEP_COST * 2  # Stronger penalty for being lost
            self.prev_action = action.copy()

            alt = self.ros_node.get_altitude()
            if alt is not None and alt < MIN_ALTITUDE:
                terminated = True
            if self.step_count >= MAX_STEPS:
                truncated = True

            info = self._build_common_info()
            info["tracking_lost"] = True
            info["tracking_lost_steps"] = self.tracking_lost_steps
            info["total_reward"] = reward
            return obs, float(reward), terminated, truncated, info

        else:
            # Tracking is healthy — update last good position
            if self.tracking_lost_steps > 0:
                print(f"[ActiveSLAMEnv] Tracking recovered after {self.tracking_lost_steps} steps")
            self.tracking_lost_steps = 0
            self.last_good_position = (current_x, current_y)

        # ----------------------------------------------------------
        # 1b. Stale data detection — silent RTAB-Map failure
        #     If covariance trace is frozen for 8+ steps, RTAB-Map has
        #     silently stopped processing but hasn't hit the tracking-lost
        #     threshold yet. Treat as tracking lost.
        # ----------------------------------------------------------
        current_cov_check = self.ros_node.get_covariance_trace()
        if self._stale_cov_trace is not None and abs(current_cov_check - self._stale_cov_trace) < 0.0001:
            self._stale_data_steps += 1
        else:
            self._stale_data_steps = 0
        self._stale_cov_trace = current_cov_check

        if self._stale_data_steps >= 8 and not self.ros_node.is_tracking_lost():
            print(f"[ActiveSLAMEnv] STALE DATA detected — cov_trace frozen at {current_cov_check:.4f} "
                  f"for {self._stale_data_steps} steps. Hovering in place.")
            # HOVER IN PLACE — do NOT fly to origin.
            # Same rationale as tracking-loss: moving during SLAM failure
            # destabilizes PX4 and causes crashes.
            self.ros_node.update_setpoint(
                current_x, current_y, flight_z_ned, yaw=self.prev_yaw)
            time.sleep(STEP_DURATION)
            self._stale_data_steps = 0  # Reset counter

            new_pos = self.ros_node.get_drone_position()
            if new_pos is not None:
                self.trajectory.append((new_pos[0], new_pos[1]))
            obs = self._get_observation()
            reward = W_STEP_COST * 1.5
            self.prev_action = action.copy()
            alt = self.ros_node.get_altitude()
            info = self._build_common_info()
            info["stale_data"] = True
            info["total_reward"] = reward
            terminated = alt is not None and alt < MIN_ALTITUDE
            truncated = self.step_count >= MAX_STEPS
            return obs, float(reward), terminated, truncated, info

        # ----------------------------------------------------------
        # 2. Covariance Panic Retreat (overrides RL action)
        # ----------------------------------------------------------
        should_retreat, retreat_wp = self._check_covariance_retreat()

        if should_retreat and retreat_wp is not None:
            target_x, target_y = retreat_wp[0], retreat_wp[1]
            flight_z_ned = retreat_wp[2]
        else:
            # ----------------------------------------------------------
            # 3. Compute raw target from RL action
            # ----------------------------------------------------------
            dx = float(action[0]) * MAX_STEP_DIST
            dy = float(action[1]) * MAX_STEP_DIST

            # Minimum movement threshold — prevent pure rotation
            move_mag = math.sqrt(dx**2 + dy**2)
            if move_mag < 0.05:
                # Action too small — hold position, don't rotate
                self.ros_node.update_setpoint(
                    current_x, current_y, flight_z_ned, yaw=self.prev_yaw)
                time.sleep(STEP_DURATION)
                new_pos = self.ros_node.get_drone_position()
                if new_pos is not None:
                    self.trajectory.append((new_pos[0], new_pos[1]))
                obs = self._get_observation()
                reward, reward_info = self._compute_reward(action, False)
                self.prev_action = action.copy()
                alt = self.ros_node.get_altitude()
                terminated = alt is not None and alt < MIN_ALTITUDE
                truncated = self.step_count >= MAX_STEPS
                info = self._build_common_info()
                info.update(reward_info)
                info["tracking_lost"] = False
                return obs, float(reward), terminated, truncated, info

            target_x = current_x + dx
            target_y = current_y + dy

            # 4. Clamp to workspace bounds
            target_x = np.clip(target_x, WS_MIN_X, WS_MAX_X)
            target_y = np.clip(target_y, WS_MIN_Y, WS_MAX_Y)

            # 5. AOU: snap to free point if target is occupied
            target_x, target_y = self._snap_to_free_point(target_x, target_y)

        # ----------------------------------------------------------
        # 6. Path validation: raytrace → A* if blocked
        # ----------------------------------------------------------
        waypoints = self._validate_and_plan_path(
            current_x, current_y, target_x, target_y)

        if len(waypoints) == 0:
            waypoints = [(current_x, current_y)]

        immediate_x, immediate_y = waypoints[0]

        # ----------------------------------------------------------
        # 7. Compute yaw toward direction of travel (rate-limited)
        # ----------------------------------------------------------
        move_dx = immediate_x - current_x
        move_dy = immediate_y - current_y
        if abs(move_dx) > 0.01 or abs(move_dy) > 0.01:
            desired_yaw = math.atan2(move_dy, move_dx)
        else:
            desired_yaw = self.prev_yaw

        # No-rotate-near-walls: if LiDAR min < 0.6m, suppress yaw changes
        # This prevents RTAB-Map tracking loss from rotation near featureless surfaces
        lidar_min = self.ros_node.get_min_lidar_range()
        if lidar_min < 0.6:
            yaw_limit = 0.05  # Almost no rotation when near walls
        else:
            yaw_limit = MAX_YAW_CHANGE

        yaw_diff = desired_yaw - self.prev_yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        yaw_diff = max(-yaw_limit, min(yaw_limit, yaw_diff))
        target_yaw = self.prev_yaw + yaw_diff
        target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi
        self.prev_yaw = target_yaw

        # ----------------------------------------------------------
        # 8. Reactive safety: check before sending setpoint
        # ----------------------------------------------------------
        if self._check_reactive_safety(target_yaw):
            immediate_x = current_x
            immediate_y = current_y
            # Revert yaw change when safety triggers hover
            self.prev_yaw = self.prev_yaw - yaw_diff
            target_yaw = self.prev_yaw

        # ----------------------------------------------------------
        # 9. Send setpoint
        # ----------------------------------------------------------
        self.ros_node.update_setpoint(
            immediate_x, immediate_y, flight_z_ned, yaw=target_yaw)

        time.sleep(STEP_DURATION)

        # ----------------------------------------------------------
        # 10. Record trajectory
        # ----------------------------------------------------------
        new_pos = self.ros_node.get_drone_position()
        if new_pos is not None:
            self.trajectory.append((new_pos[0], new_pos[1]))

        # ----------------------------------------------------------
        # 11. Check termination conditions
        # ----------------------------------------------------------
        alt = self.ros_node.get_altitude()

        if alt is not None and alt < MIN_ALTITUDE:
            terminated = True
            collision_occurred = True
            print(f"[Step {self.step_count}] CRASHED — altitude={alt:.2f}m")

        # LiDAR collision — DON'T terminate, retreat to last good position
        if self._check_lidar_collision():
            collision_occurred = True
            self.consecutive_collision_steps += 1
            if self.consecutive_collision_steps <= 3:  # Only print first few
                print(f"[Step {self.step_count}] COLLISION — LiDAR min={self.ros_node.get_min_lidar_range():.2f}m "
                      f"(consecutive: {self.consecutive_collision_steps})")

            if self.last_good_position is not None and not terminated:
                lgx, lgy = self.last_good_position
                self.ros_node.update_setpoint(lgx, lgy, flight_z_ned, yaw=self.prev_yaw)

            # NEW: Force truncation if stuck in collision loop
            if self.consecutive_collision_steps >= MAX_CONSECUTIVE_COLLISIONS:
                truncated = True
                print(f"[Step {self.step_count}] STUCK in collision loop — truncating episode")
        else:
            self.consecutive_collision_steps = 0  # Reset counter when no collision

        if new_pos is not None:
            oob = (new_pos[0] < WS_MIN_X - 1.0 or new_pos[0] > WS_MAX_X + 1.0 or
                   new_pos[1] < WS_MIN_Y - 1.0 or new_pos[1] > WS_MAX_Y + 1.0)
            if oob:
                terminated = True
                collision_occurred = True
                print(f"[Step {self.step_count}] OUT OF BOUNDS — pos=({new_pos[0]:.1f}, {new_pos[1]:.1f})")

        if self.step_count >= MAX_STEPS:
            truncated = True
            print(f"[Step {self.step_count}] MAX STEPS reached.")

        coverage = self.ros_node.known_cells_count / max(self.ros_node.total_cells, 1)
        if coverage > 0.95:
            terminated = True
            print(f"[Step {self.step_count}] COVERAGE TARGET reached: {coverage:.1%}")

        # ----------------------------------------------------------
        # 12. Altitude management
        # ----------------------------------------------------------
        if not terminated and not truncated:
            self._check_altitude_bump()

        # ----------------------------------------------------------
        # 13. Build observation and compute reward
        # ----------------------------------------------------------
        obs = self._get_observation()
        reward, reward_info = self._compute_reward(action, collision_occurred)

        self.prev_action = action.copy()

        # ----------------------------------------------------------
        # 14. Build info dict
        # ----------------------------------------------------------
        info = self._build_common_info()
        info.update(reward_info)   # reward components override common fields
        info["tracking_lost"] = self.ros_node.is_tracking_lost()

        if new_pos is not None:
            info["pos_x"] = new_pos[0]
            info["pos_y"] = new_pos[1]

        if self.step_count % 50 == 0:
            print(f"[Step {self.step_count}/{MAX_STEPS}] "
                  f"R={reward:.2f} cov={coverage:.1%} "
                  f"cov_trace={reward_info.get('cov_trace', 0):.4f} "
                  f"alt={self.current_flight_altitude:.1f}m "
                  f"frontiers={len(self.frontier_clusters)} "
                  f"near_f={self.nearest_frontier_dist:.1f}m "
                  f"new_cells={reward_info.get('new_cells', 0)}")

        return obs, float(reward), terminated, truncated, info