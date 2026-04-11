#!/usr/bin/env python3
"""
SLAMDataCollector: ROS2 node that bridges RTAB-Map/PX4 data to the RL environment.

Architecture alignment:
- /rtabmap/localization_pose → global covariance (core novelty signal)
- /rtabmap/octomap_grid      → 2D Z-sliced occupancy grid (CNN Channel 1)
- /rtabmap/octomap_full       → 3D OctoMap (volumetric info gain in reward)
- /scan                       → 2D LiDAR (collision detection + reactive safety)
- /depth_camera               → depth image (close-range reactive avoidance)
- set_z_filter()              → dynamic altitude slicing for projected map
- PX4 topics                  → drone position, status, and flight commands

Key design decisions:
- VOLATILE QoS for PX4 topics (matches PX4's DDS profile)
- RELIABLE QoS for SLAM topics (matches RTAB-Map defaults)
- BEST_EFFORT for sensor topics (LiDAR, depth) for lowest latency
- Z-filter state tracked so env can query current slice altitude
- LiDAR data stored as numpy array with angular metadata for directional queries
- Tracking loss detection via consecutive failed matches in /rtabmap/info
- Map NEVER resets — persists and grows across episodes
"""

import numpy as np
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2, LaserScan, Image
from rtabmap_msgs.msg import Info
from octomap_msgs.msg import Octomap

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


class SLAMDataCollector(Node):
    """Collects all SLAM, sensor, and PX4 data asynchronously for the RL environment."""

    def __init__(self):
        super().__init__("slam_data_collector")

        # === QoS Profiles ===
        # RELIABLE for SLAM topics (RTAB-Map uses RELIABLE by default)
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # BEST_EFFORT + VOLATILE for PX4 topics (matches PX4's QoS)
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # BEST_EFFORT for high-frequency sensor topics (LiDAR, depth)
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ============================================
        # SLAM Subscriptions
        # ============================================
        self.odom_sub = self.create_subscription(
            Odometry, "/rtabmap/odom", self._odom_cb, qos_reliable)

        self.loc_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/rtabmap/localization_pose",
            self._loc_pose_cb, qos_reliable)

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/rtabmap/octomap_grid",
            self._map_cb, qos_reliable)

        self.octomap_sub = self.create_subscription(
            Octomap, "/rtabmap/octomap_full",
            self._octomap_cb, qos_reliable)

        self.frontier_sub = self.create_subscription(
            PointCloud2, "/rtabmap/octomap_global_frontier_space",
            self._frontier_cb, qos_reliable)

        self.info_sub = self.create_subscription(
            Info, "/rtabmap/info",
            self._info_cb, qos_reliable)

        # ============================================
        # Sensor Subscriptions (LiDAR + Depth Camera)
        # ============================================
        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan",
            self._lidar_cb, qos_sensor)

        self.depth_sub = self.create_subscription(
            Image, "/depth_camera",
            self._depth_cb, qos_sensor)

        # ============================================
        # PX4 Subscriptions
        # ============================================
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self._local_pos_cb, qos_px4)
        self.status_sub = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._status_cb, qos_px4)

        # ============================================
        # PX4 Publishers
        # ============================================
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_px4)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_px4)
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_px4)

        # ============================================
        # SLAM Data Storage
        # ============================================
        self.odom = None
        self.occupancy_grid = None
        self.grid_info = None
        self.global_pose_covariance = np.zeros(36, dtype=np.float64)
        self.frontier_count = 0
        self.frontier_points = None
        self.loop_closure_id = 0
        self.proximity_detection_id = 0
        self.octomap_data = None

        # Map tracking
        self.known_cells_count = 0
        self.total_cells = 0

        # Tracking quality — monitors RTAB-Map visual odometry health
        self.tracking_lost = False
        self._consecutive_no_match = 0

        # ============================================
        # LiDAR Data Storage
        # ============================================
        self.lidar_ranges = None
        self.lidar_min_range = float('inf')
        self.lidar_angle_min = 0.0
        self.lidar_angle_max = 0.0
        self.lidar_angle_increment = 0.0
        self.lidar_range_min = 0.0
        self.lidar_range_max = 0.0

        # ============================================
        # Depth Camera Data Storage
        # ============================================
        self.depth_min_distance = float('inf')
        self.depth_image = None

        # ============================================
        # PX4 Data Storage
        # ============================================
        self.local_position = None
        self.vehicle_status = None

        # ============================================
        # Z-filter State (for altitude-sliced maps)
        # ============================================
        self.set_params_client = self.create_client(
            SetParameters, "/rtabmap/rtabmap/set_parameters")
        self.z_filter_min = 0.0
        self.z_filter_max = 0.0
        self.z_filter_active = False

        self.get_logger().info("SLAM data collector initialized (full architecture)")
        self.get_logger().info("  Covariance: /rtabmap/localization_pose (global)")
        self.get_logger().info("  3D map:     /rtabmap/octomap_full")
        self.get_logger().info("  2D map:     /rtabmap/octomap_grid (Z-filterable)")
        self.get_logger().info("  LiDAR:      /scan (360° collision detection)")
        self.get_logger().info("  Depth:      /depth_camera (reactive avoidance)")

        # ============================================
        # Continuous Offboard Heartbeat Timer
        # ============================================
        self._offboard_active = False
        self._current_setpoint = [0.0, 0.0, -1.5, 0.0]
        self._heartbeat_timer = self.create_timer(0.1, self._heartbeat_timer_cb)

    def _heartbeat_timer_cb(self):
        """10Hz heartbeat timer — keeps offboard mode alive continuously."""
        if not self._offboard_active:
            return
        self.publish_offboard_heartbeat()
        x, y, z, yaw = self._current_setpoint
        self.publish_setpoint(x, y, z, yaw=yaw)

    def start_offboard_stream(self, x=0.0, y=0.0, z=-1.5, yaw=0.0):
        """Start continuous offboard heartbeat + setpoint streaming."""
        self._current_setpoint = [float(x), float(y), float(z), float(yaw)]
        self._offboard_active = True
        self.get_logger().debug("Offboard stream started")

    def update_setpoint(self, x, y, z, yaw=0.0):
        """Update the continuously-streamed setpoint."""
        self._current_setpoint = [float(x), float(y), float(z), float(yaw)]

    def stop_offboard_stream(self):
        """Stop the continuous offboard heartbeat (before landing/disarm)."""
        self._offboard_active = False
        self.get_logger().debug("Offboard stream stopped")

    # ============================================
    # SLAM Callbacks
    # ============================================
    def _odom_cb(self, msg):
        self.odom = msg

    def _loc_pose_cb(self, msg):
        """Global localization pose — covariance DECREASES after loop closures."""
        self.global_pose_covariance = np.array(msg.pose.covariance, dtype=np.float64)

    def _map_cb(self, msg):
        """2D projected occupancy grid from OctoMap."""
        w, h = msg.info.width, msg.info.height
        self.grid_info = msg.info
        raw = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.occupancy_grid = raw
        self.total_cells = w * h
        self.known_cells_count = int(np.sum(raw >= 0))

    def _octomap_cb(self, msg):
        self.octomap_data = msg

    def _frontier_cb(self, msg):
        self.frontier_count = msg.width
        self.frontier_points = msg

    def _info_cb(self, msg):
        """SLAM info: loop_closure_id != 0 means loop closure detected.
        Also monitors tracking quality via consecutive failed matches."""
        self.loop_closure_id = msg.loop_closure_id
        self.proximity_detection_id = msg.proximity_detection_id

        # Track visual odometry health.
        # When RTAB-Map can't match consecutive frames, both IDs stay 0.
        # 30Hz callback × 1.5 seconds = ~45 consecutive failures → lost.
        # Faster detection (was 90/3s) prevents map corruption.
        if msg.loop_closure_id == 0 and self.proximity_detection_id == 0:
            self._consecutive_no_match += 1
        else:
            self._consecutive_no_match = 0
            if self.tracking_lost:
                self.tracking_lost = False
                self.get_logger().info("RTAB-Map tracking RECOVERED")

        if self._consecutive_no_match > 45 and not self.tracking_lost:
            self.tracking_lost = True
            self.get_logger().warn("RTAB-Map tracking LOST — will retreat")

    # ============================================
    # Sensor Callbacks (LiDAR + Depth)
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
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w)).astype(np.float32) / 1000.0
            else:
                return

            self.depth_image = depth
            cy, cx = h // 4, w // 4
            center = depth[cy:3*cy, cx:3*cx]
            valid = np.isfinite(center) & (center > 0.1) & (center < 20.0)
            if np.any(valid):
                self.depth_min_distance = float(np.min(center[valid]))
            else:
                self.depth_min_distance = float('inf')
        except Exception:
            pass

    # ============================================
    # PX4 Callbacks
    # ============================================
    def _local_pos_cb(self, msg):
        self.local_position = msg

    def _status_cb(self, msg):
        self.vehicle_status = msg

    # ============================================
    # PX4 Command Methods
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
    # Data Accessor Methods — Position & Covariance
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
        """Returns True if RTAB-Map has lost visual odometry tracking."""
        return self.tracking_lost

    # ============================================
    # Data Accessor Methods — LiDAR
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
    # ============================================
    def get_depth_min_distance(self):
        return self.depth_min_distance

    # ============================================
    # Z-Filter for Altitude Slicing
    # ============================================
    def set_z_filter(self, min_z, max_z):
        if not self.set_params_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("RTAB-Map parameter service not available for Z-filter")
            return False

        params = [
            Parameter(
                name="Grid/MinGroundHeight",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=str(min_z))),
            Parameter(
                name="Grid/MaxObstacleHeight",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=str(max_z))),
        ]

        request = SetParameters.Request(parameters=params)
        future = self.set_params_client.call_async(request)

        self.z_filter_min = min_z
        self.z_filter_max = max_z
        self.z_filter_active = True

        self.get_logger().info(f"Z-filter set: [{min_z:.2f}m, {max_z:.2f}m]")
        return True

    def get_z_filter_state(self):
        return self.z_filter_min, self.z_filter_max, self.z_filter_active

    # ============================================
    # Frontier Point Extraction (3D → 2D)
    # ============================================
    def get_frontier_points_2d(self, z_min=None, z_max=None):
        if self.frontier_points is None:
            return np.empty((0, 2), dtype=np.float32)

        msg = self.frontier_points
        if msg.width == 0:
            return np.empty((0, 2), dtype=np.float32)

        try:
            x_off = y_off = z_off = None
            for field in msg.fields:
                if field.name == 'x':
                    x_off = field.offset
                elif field.name == 'y':
                    y_off = field.offset
                elif field.name == 'z':
                    z_off = field.offset

            if x_off is None or y_off is None:
                return np.empty((0, 2), dtype=np.float32)

            point_step = msg.point_step
            data = msg.data
            n_points = msg.width * msg.height
            points = []

            for i in range(n_points):
                base = i * point_step
                x = struct.unpack_from('f', data, base + x_off)[0]
                y = struct.unpack_from('f', data, base + y_off)[0]

                if z_off is not None and z_min is not None and z_max is not None:
                    z = struct.unpack_from('f', data, base + z_off)[0]
                    if z < z_min or z > z_max:
                        continue

                points.append([x, y])

            if len(points) == 0:
                return np.empty((0, 2), dtype=np.float32)
            return np.array(points, dtype=np.float32)

        except Exception:
            return np.empty((0, 2), dtype=np.float32)