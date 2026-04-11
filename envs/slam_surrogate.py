#!/usr/bin/env python3
"""
SLAMSurrogate: Drop-in replacement for SLAMDataCollector that eliminates RTAB-Map.

PURPOSE:
  Replaces RTAB-Map's fragile visual SLAM pipeline with ground-truth-based
  equivalents for RL training. Provides the EXACT same public API as
  SLAMDataCollector so that active_slam_env.py needs only a one-line
  import change.

WHAT CHANGES:
  - Occupancy grid: built from depth camera + ground-truth PX4 pose
  - Covariance trace: synthetic model calibrated to match RTAB-Map behavior
  - Loop closures: distance-based revisit detection
  - Tracking loss: never happens (ground-truth pose doesn't lose tracking)
  - No rtabmap.db, no SQLite, no corruption, no restarts

WHAT STAYS THE SAME:
  - All PX4 subscriptions, publishers, heartbeat, arm/disarm/land
  - LiDAR subscription and all LiDAR accessor methods
  - Depth camera subscription
  - Every public method signature (100% API compatible)

USAGE:
  In active_slam_env.py, change:
    from envs.slam_collector import SLAMDataCollector
  to:
    from envs.slam_surrogate import SLAMSurrogate as SLAMDataCollector

  Everything else stays identical.

Author: Generated for Active SLAM RL training pipeline
"""

import math
import numpy as np
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import MapMetaData
from sensor_msgs.msg import LaserScan, Image

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


# ============================================================
# Synthetic Covariance Model Parameters
# ============================================================
# These control how the synthetic covariance behaves.
# Calibrated to approximate RTAB-Map's covariance in the warehouse.
#
# The model: cov grows linearly with distance traveled since last
# "loop closure" (revisit), and drops sharply on revisit events.
# Gaussian noise adds realism.
#
COV_DRIFT_RATE = 0.015        # Covariance growth per meter traveled
COV_BASE_NOISE = 0.001        # Minimum covariance (sensor noise floor)
COV_LOOP_CLOSURE_DROP = 0.7   # Multiplicative drop on loop closure (0.7 = 30% reduction)
COV_MAX_CLAMP = 5.0           # Maximum synthetic covariance
COV_NOISE_STD = 0.002         # Gaussian noise added each step

# ============================================================
# Loop Closure Detection Parameters
# ============================================================
LC_REVISIT_RADIUS = 1.0       # Must return within this distance of a past pose
LC_MIN_TRAVEL_DIST = 8.0      # Must have traveled at least this far since last visit
                               # Was 3.0 — too low, triggered from takeoff wobble
LC_SAMPLE_INTERVAL = 1.0      # Store a pose sample every 1.0m of travel
                               # Was 0.5 — stored too many poses near origin

# ============================================================
# Occupancy Grid Parameters
# ============================================================
GRID_RESOLUTION = 0.05        # 5cm per cell (matches typical RTAB-Map output)
GRID_ORIGIN_X = -12.0         # Grid origin in world frame (meters)
GRID_ORIGIN_Y = -12.0         # Generous padding around warehouse
GRID_WIDTH = 480              # 480 cells × 0.05m = 24m coverage
GRID_HEIGHT = 480             # 480 cells × 0.05m = 24m coverage

# Depth camera projection parameters (OakD-Lite on x500_depth)
DEPTH_FOV_H = 70.0 * math.pi / 180.0   # 70° horizontal FOV
DEPTH_MAX_RANGE = 4.0                    # Max usable depth range (meters)
                                         # RTAB-Map gets useful depth to ~3-4m with OakD-Lite
                                         # Was 8.0 — mapped too aggressively
DEPTH_MIN_RANGE = 0.2                    # Min usable depth range
DEPTH_NUM_RAYS = 32                      # Number of rays to cast per update
                                         # Was 64 — reduced for more realistic mapping speed


class SLAMSurrogate(Node):
    """Ground-truth SLAM surrogate — API-compatible with SLAMDataCollector.

    Builds occupancy grids from depth camera data using PX4's ground-truth
    pose. Provides synthetic covariance and loop closure signals that
    approximate RTAB-Map's behavior without any of the fragility.
    """

    def __init__(self):
        super().__init__("slam_surrogate")

        # === QoS Profiles (identical to SLAMDataCollector) ===
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ============================================
        # Sensor Subscriptions (KEPT — real Gazebo data)
        # ============================================
        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan",
            self._lidar_cb, qos_sensor)

        self.depth_sub = self.create_subscription(
            Image, "/depth_camera",
            self._depth_cb, qos_sensor)

        # ============================================
        # PX4 Subscriptions (KEPT — identical)
        # ============================================
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self._local_pos_cb, qos_px4)
        self.status_sub = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._status_cb, qos_px4)

        # ============================================
        # PX4 Publishers (KEPT — identical)
        # ============================================
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_px4)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_px4)
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_px4)

        # ============================================
        # Ground-Truth Occupancy Grid (REPLACES RTAB-Map)
        # ============================================
        # -1 = unknown, 0 = free, 100 = occupied
        self._raw_grid = np.full(
            (GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)

        # Public grid interface (matches SLAMDataCollector exactly)
        self.occupancy_grid = self._raw_grid
        self.grid_info = self._make_grid_info()
        self.total_cells = GRID_WIDTH * GRID_HEIGHT
        self.known_cells_count = 0

        # ============================================
        # Synthetic Covariance State (REPLACES RTAB-Map)
        # ============================================
        self.global_pose_covariance = np.zeros(36, dtype=np.float64)
        self._cov_trace = COV_BASE_NOISE
        self._distance_since_lc = 0.0   # Distance traveled since last loop closure
        self._prev_pose_for_dist = None  # For incremental distance tracking

        # ============================================
        # Loop Closure Detection (REPLACES RTAB-Map)
        # ============================================
        self.loop_closure_id = 0
        self.proximity_detection_id = 0
        self._lc_counter = 0             # Incrementing ID for loop closures
        self._pose_history = []          # [(x, y, distance_at_sample)]
        self._total_distance = 0.0       # Odometer
        self._last_sample_distance = 0.0 # Distance at last pose sample

        # ============================================
        # Tracking (ALWAYS healthy — no RTAB-Map to lose)
        # ============================================
        self.tracking_lost = False
        self._consecutive_no_match = 0

        # ============================================
        # Frontier points (computed from grid, not subscribed)
        # ============================================
        self.frontier_count = 0
        self.frontier_points = None      # Not used — env computes its own

        # ============================================
        # Odom / OctoMap stubs (for API completeness)
        # ============================================
        self.odom = None
        self.octomap_data = None

        # ============================================
        # LiDAR Data Storage (identical to SLAMDataCollector)
        # ============================================
        self.lidar_ranges = None
        self.lidar_min_range = float('inf')
        self.lidar_angle_min = 0.0
        self.lidar_angle_max = 0.0
        self.lidar_angle_increment = 0.0
        self.lidar_range_min = 0.0
        self.lidar_range_max = 0.0

        # ============================================
        # Depth Camera Data Storage (identical to SLAMDataCollector)
        # ============================================
        self.depth_min_distance = float('inf')
        self.depth_image = None

        # ============================================
        # PX4 Data Storage (identical to SLAMDataCollector)
        # ============================================
        self.local_position = None
        self.vehicle_status = None

        # ============================================
        # Z-filter state (stub — not needed without RTAB-Map)
        # ============================================
        self.z_filter_min = 0.0
        self.z_filter_max = 0.0
        self.z_filter_active = False

        # ============================================
        # Offboard Heartbeat (identical to SLAMDataCollector)
        # ============================================
        self._offboard_active = False
        self._current_setpoint = [0.0, 0.0, -1.5, 0.0]
        self._heartbeat_timer = self.create_timer(0.1, self._heartbeat_timer_cb)

        # ============================================
        # Grid Update Timer (replaces RTAB-Map processing)
        # Runs at 2Hz — integrates depth into occupancy grid
        # Was 5Hz but mapped too fast compared to real RTAB-Map
        # ============================================
        self._grid_update_timer = self.create_timer(0.5, self._grid_update_cb)

        self.get_logger().info("SLAM Surrogate initialized (ground-truth mode)")
        self.get_logger().info("  Occupancy: depth camera + GT pose (no RTAB-Map)")
        self.get_logger().info("  Covariance: synthetic model")
        self.get_logger().info("  Loop closure: distance-based revisit detection")
        self.get_logger().info("  LiDAR: /scan (real Gazebo sensor)")
        self.get_logger().info("  Depth: /depth_camera (real Gazebo sensor)")

    # ============================================
    # Grid Metadata
    # ============================================
    def _make_grid_info(self):
        """Create a MapMetaData object matching the grid parameters."""
        info = MapMetaData()
        info.resolution = GRID_RESOLUTION
        info.width = GRID_WIDTH
        info.height = GRID_HEIGHT
        info.origin.position.x = float(GRID_ORIGIN_X)
        info.origin.position.y = float(GRID_ORIGIN_Y)
        info.origin.position.z = 0.0
        info.origin.orientation.w = 1.0
        return info

    # ============================================
    # Heartbeat Timer (identical to SLAMDataCollector)
    # ============================================
    def _heartbeat_timer_cb(self):
        if not self._offboard_active:
            return
        self.publish_offboard_heartbeat()
        x, y, z, yaw = self._current_setpoint
        self.publish_setpoint(x, y, z, yaw=yaw)

    # ============================================
    # Grid Update — integrates depth into occupancy grid
    # This is the CORE replacement for RTAB-Map's mapping
    # ============================================
    def _grid_update_cb(self):
        """Project depth image into 2D occupancy grid using GT pose.

        Runs at 2Hz. Uses ONLY the depth camera (70° FOV, 4m range) to
        build the map — matching RTAB-Map's visual-only mapping behavior.
        LiDAR is NOT used for mapping (only collision avoidance).
        """
        pos = self.get_drone_position()
        if pos is None or self.depth_image is None:
            return

        drone_x, drone_y, drone_z = pos[0], pos[1], pos[2]
        drone_alt = -drone_z  # NED to altitude
        yaw = self.get_drone_yaw()

        depth = self.depth_image
        h, w = depth.shape

        # Cast rays across the horizontal FOV
        for i in range(DEPTH_NUM_RAYS):
            # Angle of this ray relative to camera center
            frac = (i / max(DEPTH_NUM_RAYS - 1, 1)) - 0.5  # [-0.5, 0.5]
            ray_angle = yaw + frac * DEPTH_FOV_H

            # Sample depth at corresponding column
            col = int((frac + 0.5) * w)
            col = max(0, min(w - 1, col))

            # Take median of a vertical strip for robustness
            strip = depth[h // 4:3 * h // 4, max(0, col - 2):min(w, col + 3)]
            valid = strip[np.isfinite(strip) & (strip > DEPTH_MIN_RANGE) & (strip < DEPTH_MAX_RANGE)]

            if len(valid) == 0:
                # No valid depth — SKIP this ray entirely
                # Was marking 8m of free space, which mapped way too aggressively
                # RTAB-Map only maps what it can actually see with features
                continue

            ray_dist = float(np.median(valid))
            hit_obstacle = True

            # Project ray along the ground plane (ignore vertical component)
            cos_a = math.cos(ray_angle)
            sin_a = math.sin(ray_angle)

            # Mark cells along the ray as free
            step_size = GRID_RESOLUTION * 0.8  # Slightly less than cell size
            n_steps = int(ray_dist / step_size)

            for s in range(n_steps):
                d = s * step_size
                wx = drone_x + d * cos_a
                wy = drone_y + d * sin_a
                gx, gy = self._world_to_grid_internal(wx, wy)
                if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
                    self._raw_grid[gy, gx] = 0  # Free

            # Mark the endpoint as occupied (if we actually hit something)
            if hit_obstacle:
                wx = drone_x + ray_dist * cos_a
                wy = drone_y + ray_dist * sin_a
                gx, gy = self._world_to_grid_internal(wx, wy)
                if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
                    self._raw_grid[gy, gx] = 100  # Occupied

        # NOTE: LiDAR is NOT used for map building — only for collision avoidance.
        # RTAB-Map is a VISUAL SLAM system that maps using the camera, not LiDAR.
        # Using LiDAR (360°, 10m range) would map the entire warehouse instantly,
        # making the exploration problem trivial and preventing the RL agent
        # from learning anything useful.
        # The LiDAR data is still received and used by active_slam_env.py for
        # reactive safety checks (get_min_lidar_range, get_lidar_range_in_direction).

        # Update public grid reference and cell counts
        self.occupancy_grid = self._raw_grid
        self.known_cells_count = int(np.sum(self._raw_grid >= 0))

        # Update covariance and loop closure models
        self._update_covariance_model(drone_x, drone_y)
        self._check_loop_closure(drone_x, drone_y)

    def _integrate_lidar(self, drone_x, drone_y, drone_yaw):
        """Integrate 2D LiDAR scan into occupancy grid.

        The LiDAR has ~270° FOV and is much better for building 2D maps
        than the depth camera alone.
        """
        if self.lidar_ranges is None or self.lidar_angle_increment == 0.0:
            return

        ranges = self.lidar_ranges
        n = len(ranges)
        step_size = GRID_RESOLUTION * 0.8

        for i in range(0, n, 2):  # Skip every other ray for performance
            r = ranges[i]
            if not np.isfinite(r) or r < self.lidar_range_min or r > self.lidar_range_max:
                continue

            # Angle in world frame
            angle = self.lidar_angle_min + i * self.lidar_angle_increment + drone_yaw
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Mark free cells along the ray
            n_steps = int(r / step_size)
            for s in range(n_steps):
                d = s * step_size
                wx = drone_x + d * cos_a
                wy = drone_y + d * sin_a
                gx, gy = self._world_to_grid_internal(wx, wy)
                if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
                    self._raw_grid[gy, gx] = 0

            # Mark endpoint as occupied
            wx = drone_x + r * cos_a
            wy = drone_y + r * sin_a
            gx, gy = self._world_to_grid_internal(wx, wy)
            if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
                self._raw_grid[gy, gx] = 100

    def _world_to_grid_internal(self, wx, wy):
        """Convert world coordinates to grid indices."""
        gx = int((wx - GRID_ORIGIN_X) / GRID_RESOLUTION)
        gy = int((wy - GRID_ORIGIN_Y) / GRID_RESOLUTION)
        return gx, gy

    # ============================================
    # Synthetic Covariance Model
    # ============================================
    def _update_covariance_model(self, x, y):
        """Update synthetic covariance based on distance traveled.

        Model:
          - Covariance grows linearly with distance from last loop closure
          - Gaussian noise added for realism
          - Drops on loop closure events
          - Clamped to [COV_BASE_NOISE, COV_MAX_CLAMP]

        This approximates RTAB-Map's behavior where covariance grows as
        the drone explores new areas and shrinks when loop closures
        refine the pose graph.
        """
        if self._prev_pose_for_dist is not None:
            dx = x - self._prev_pose_for_dist[0]
            dy = y - self._prev_pose_for_dist[1]
            step_dist = math.sqrt(dx * dx + dy * dy)
            # Only count as travel if drone actually moved meaningfully
            # Filters out takeoff wobble, hover oscillations, and GPS jitter
            # Without this, tiny oscillations accumulate into fake "travel"
            # which triggers premature loop closures
            if step_dist > 0.05:
                self._distance_since_lc += step_dist
                self._total_distance += step_dist
                self._prev_pose_for_dist = (x, y)
        else:
            self._prev_pose_for_dist = (x, y)

        # Covariance = base + drift * distance + noise
        self._cov_trace = (
            COV_BASE_NOISE
            + COV_DRIFT_RATE * self._distance_since_lc
            + np.random.normal(0, COV_NOISE_STD)
        )
        self._cov_trace = float(np.clip(self._cov_trace, COV_BASE_NOISE, COV_MAX_CLAMP))

        # Write into the 6×6 covariance matrix diagonal (x, y, z)
        self.global_pose_covariance[0] = self._cov_trace / 3.0   # xx
        self.global_pose_covariance[7] = self._cov_trace / 3.0   # yy
        self.global_pose_covariance[14] = self._cov_trace / 3.0  # zz

    # ============================================
    # Loop Closure Detection
    # ============================================
    def _check_loop_closure(self, x, y):
        """Detect loop closures based on revisiting previous locations.

        A loop closure is triggered when:
          1. The drone is within LC_REVISIT_RADIUS of a previous pose sample
          2. It has traveled at least LC_MIN_TRAVEL_DIST since that sample
             was recorded (to prevent trivially detecting nearby poses)

        On detection:
          - loop_closure_id is set to a new unique ID
          - Covariance drops by COV_LOOP_CLOSURE_DROP factor
          - Distance-since-LC counter resets
        """
        # Store pose samples at regular distance intervals
        if (self._total_distance - self._last_sample_distance) >= LC_SAMPLE_INTERVAL:
            self._pose_history.append((x, y, self._total_distance))
            self._last_sample_distance = self._total_distance

        # Check for revisits
        # Only check older poses (not the last few we just recorded)
        n_skip = max(0, len(self._pose_history) - int(LC_MIN_TRAVEL_DIST / LC_SAMPLE_INTERVAL) - 5)
        for i in range(n_skip):
            px, py, pdist = self._pose_history[i]
            spatial_dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            travel_dist = self._total_distance - pdist

            if spatial_dist < LC_REVISIT_RADIUS and travel_dist > LC_MIN_TRAVEL_DIST:
                # Loop closure detected!
                self._lc_counter += 1
                self.loop_closure_id = self._lc_counter

                # Reduce covariance
                self._cov_trace *= COV_LOOP_CLOSURE_DROP
                self._distance_since_lc = 0.0

                # Remove the matched pose so we don't re-trigger
                self._pose_history.pop(i)

                self.get_logger().info(
                    f"Loop closure #{self._lc_counter} — revisited "
                    f"({px:.1f}, {py:.1f}) after {travel_dist:.1f}m travel"
                )
                return

    # ============================================
    # Sensor Callbacks (identical to SLAMDataCollector)
    # ============================================
    def _lidar_cb(self, msg):
        self.lidar_angle_min = msg.angle_min
        self.lidar_angle_max = msg.angle_max
        self.lidar_angle_increment = msg.angle_increment
        self.lidar_range_min = msg.range_min
        self.lidar_range_max = msg.range_max

        ranges = np.array(msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        self.lidar_ranges = ranges
        if np.any(valid):
            self.lidar_min_range = float(np.min(ranges[valid]))
        else:
            self.lidar_min_range = float('inf')

    def _depth_cb(self, msg):
        try:
            h, w = msg.height, msg.width
            if msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w))
            elif msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    (h, w)).astype(np.float32) / 1000.0
            else:
                return

            self.depth_image = depth
            cy, cx = h // 4, w // 4
            center = depth[cy:3 * cy, cx:3 * cx]
            valid = np.isfinite(center) & (center > 0.1) & (center < 20.0)
            if np.any(valid):
                self.depth_min_distance = float(np.min(center[valid]))
            else:
                self.depth_min_distance = float('inf')
        except Exception:
            pass

    # ============================================
    # PX4 Callbacks (identical to SLAMDataCollector)
    # ============================================
    def _local_pos_cb(self, msg):
        self.local_position = msg

    def _status_cb(self, msg):
        self.vehicle_status = msg

    # ============================================
    # PX4 Command Methods (identical to SLAMDataCollector)
    # ============================================
    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, x, y, z, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def publish_command(self, command, **kwargs):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = kwargs.get("param1", 0.0)
        msg.param2 = kwargs.get("param2", 0.0)
        msg.param3 = kwargs.get("param3", 0.0)
        msg.param4 = kwargs.get("param4", 0.0)
        msg.param5 = kwargs.get("param5", 0.0)
        msg.param6 = kwargs.get("param6", 0.0)
        msg.param7 = kwargs.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def arm(self):
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("ARM command sent")

    def disarm(self):
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info("DISARM command sent")

    def engage_offboard(self):
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("OFFBOARD mode command sent")

    def land(self):
        self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("LAND command sent")

    # ============================================
    # Offboard Stream (identical to SLAMDataCollector)
    # ============================================
    def start_offboard_stream(self, x=0.0, y=0.0, z=-1.5, yaw=0.0):
        self._current_setpoint = [float(x), float(y), float(z), float(yaw)]
        self._offboard_active = True
        self.get_logger().debug("Offboard stream started")

    def update_setpoint(self, x, y, z, yaw=0.0):
        self._current_setpoint = [float(x), float(y), float(z), float(yaw)]

    def stop_offboard_stream(self):
        self._offboard_active = False
        self.get_logger().debug("Offboard stream stopped")

    # ============================================
    # Data Accessor Methods — Position & Covariance
    # (identical signatures to SLAMDataCollector)
    # ============================================
    def get_drone_position(self):
        if self.local_position is None:
            return None
        return np.array([
            self.local_position.x,
            self.local_position.y,
            self.local_position.z,
        ], dtype=np.float64)

    def get_drone_yaw(self):
        if self.local_position is None:
            return 0.0
        return float(self.local_position.heading)

    def get_covariance_trace(self):
        diag_indices = [0, 7, 14]
        return float(sum(self.global_pose_covariance[i] for i in diag_indices))

    def get_covariance_trace_normalized(self, max_val=1.0):
        raw = self.get_covariance_trace()
        return float(np.clip(raw, 0.0, max_val) / max_val)

    def is_armed(self):
        if self.vehicle_status is None:
            return False
        return self.vehicle_status.arming_state == 2

    def get_altitude(self):
        pos = self.get_drone_position()
        if pos is None:
            return None
        return float(-pos[2])

    def is_tracking_lost(self):
        """Always returns False — ground-truth pose never loses tracking."""
        return False

    # ============================================
    # Data Accessor Methods — LiDAR
    # (identical to SLAMDataCollector)
    # ============================================
    def get_min_lidar_range(self):
        return self.lidar_min_range

    def get_lidar_range_in_direction(self, target_angle_rad):
        if self.lidar_ranges is None or self.lidar_angle_increment == 0.0:
            return float('inf')

        cone_half = 0.2618
        ranges = self.lidar_ranges
        n = len(ranges)

        min_range = float('inf')
        for i in range(n):
            angle = self.lidar_angle_min + i * self.lidar_angle_increment
            diff = (angle - target_angle_rad + np.pi) % (2 * np.pi) - np.pi
            if abs(diff) <= cone_half:
                r = ranges[i]
                if np.isfinite(r) and self.lidar_range_min <= r <= self.lidar_range_max:
                    min_range = min(min_range, r)

        return float(min_range)

    def get_lidar_ranges_array(self):
        return self.lidar_ranges

    # ============================================
    # Data Accessor Methods — Depth Camera
    # (identical to SLAMDataCollector)
    # ============================================
    def get_depth_min_distance(self):
        return self.depth_min_distance

    # ============================================
    # Z-Filter Stub (no-op without RTAB-Map)
    # ============================================
    def set_z_filter(self, min_z, max_z):
        """No-op — Z-filtering was an RTAB-Map OctoMap feature.
        The ground-truth grid is already 2D-projected."""
        self.z_filter_min = min_z
        self.z_filter_max = max_z
        self.z_filter_active = True
        self.get_logger().info(
            f"Z-filter set (stub): [{min_z:.2f}m, {max_z:.2f}m]")
        return True

    def get_z_filter_state(self):
        return self.z_filter_min, self.z_filter_max, self.z_filter_active

    # ============================================
    # Frontier Points (stub — env computes its own)
    # ============================================
    def get_frontier_points_2d(self, z_min=None, z_max=None):
        """Return empty — the environment computes frontiers itself
        via _detect_frontiers() from the occupancy grid."""
        return np.empty((0, 2), dtype=np.float32)

    # ============================================
    # Grid Reset (for episode transitions)
    # ============================================
    def reset_grid(self):
        """Reset the occupancy grid for a fresh mapping session.

        Call this if you want episodes to start with a clean map.
        If you want the map to persist across episodes (like RTAB-Map),
        don't call this.
        """
        self._raw_grid = np.full(
            (GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)
        self.occupancy_grid = self._raw_grid
        self.known_cells_count = 0

        # Reset covariance and loop closure state
        self._cov_trace = COV_BASE_NOISE
        self._distance_since_lc = 0.0
        self._prev_pose_for_dist = None
        self._pose_history = []
        self._total_distance = 0.0
        self._last_sample_distance = 0.0
        self.loop_closure_id = 0
        self._lc_counter = 0

        self.global_pose_covariance = np.zeros(36, dtype=np.float64)
        self.get_logger().info("Grid and SLAM state reset")