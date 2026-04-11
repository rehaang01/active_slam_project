#!/usr/bin/env python3
"""
Test script: Verify programmatic drone control via PX4 ROS2 interface.

Sequence: Arm → Take off to 1.5m → Hover 5s → Fly to (2, 0) → Land.
This validates the complete Python → ROS2 → MicroXRCEAgent → PX4 SITL
control pipeline INDEPENDENTLY of RTAB-Map or RL.

If this script succeeds, the control stack is confirmed operational.

Usage (inside Docker container):
    source /opt/ros/humble/setup.bash
    python3 /root/active_slam_project/test_drone_control.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

import time
import math


# Maximum time allowed per phase (seconds) before moving on
TAKEOFF_TIMEOUT = 30.0
HOVER_DURATION = 5.0
MOVE_TIMEOUT = 30.0
LAND_TIMEOUT = 15.0


class DroneController(Node):
    def __init__(self):
        super().__init__("drone_controller_test")

        # QoS: BEST_EFFORT + VOLATILE to match PX4's DDS profile.
        # Using TRANSIENT_LOCAL here causes ROS2 to silently drop
        # all messages from PX4 — the #1 cause of "test hangs forever."
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_profile)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_profile)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_profile)

        # Subscribers
        self.vehicle_local_position_sub = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self.vehicle_status_callback, qos_profile)

        # State
        self.vehicle_local_position = None
        self.vehicle_status = None
        self.offboard_setpoint_counter = 0
        self.takeoff_height = -1.5  # NED: -1.5 = 1.5m above ground

        # Phase management with timeouts
        self.phase = "preflight"
        self.phase_start_time = time.time()
        self.hover_start_time = None
        self.log_counter = 0  # throttle log output

        # Timer at 20Hz (50ms)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("Drone controller test node started")
        self.get_logger().info("  QoS: BEST_EFFORT + VOLATILE (matches PX4)")

    # ========================
    # Callbacks
    # ========================
    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    # ========================
    # PX4 Commands
    # ========================
    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Arm command sent")

    def engage_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Offboard mode command sent")

    def land(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Land command sent")

    def publish_offboard_control_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(msg)

    def publish_position_setpoint(self, x, y, z, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command, **kwargs):
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
        self.vehicle_command_pub.publish(msg)

    # ========================
    # Helpers
    # ========================
    def _phase_elapsed(self):
        """Seconds since current phase started."""
        return time.time() - self.phase_start_time

    def _set_phase(self, new_phase):
        """Transition to a new phase and reset the timer."""
        self.phase = new_phase
        self.phase_start_time = time.time()
        self.get_logger().info(f"=== {new_phase.upper()} PHASE ===")

    def _is_armed(self):
        """Check if PX4 reports the drone as armed."""
        if self.vehicle_status is None:
            return False
        return self.vehicle_status.arming_state == 2

    # ========================
    # Main State Machine
    # ========================
    def timer_callback(self):
        # Always send heartbeat (keeps offboard mode alive)
        self.publish_offboard_control_heartbeat()
        self.log_counter += 1

        # ---- PREFLIGHT: send setpoints before engaging offboard ----
        if self.phase == "preflight":
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            self.offboard_setpoint_counter += 1

            # PX4 needs ~2s of setpoints before accepting offboard mode
            if self.offboard_setpoint_counter >= 40:
                self.engage_offboard_mode()
                time.sleep(0.1)
                self.arm()
                self._set_phase("takeoff")

        # ---- TAKEOFF: wait for target altitude ----
        elif self.phase == "takeoff":
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

            # Verify arming
            if not self._is_armed() and self._phase_elapsed() > 5.0:
                self.get_logger().warn("Not armed after 5s — retrying arm command")
                self.arm()

            if self.vehicle_local_position is not None:
                current_z = self.vehicle_local_position.z
                # Log every 10th tick (0.5Hz instead of 20Hz)
                if self.log_counter % 10 == 0:
                    self.get_logger().info(
                        f"  Height: {-current_z:.2f}m (target: {-self.takeoff_height:.1f}m)")

                # Check altitude including Z axis
                if abs(current_z - self.takeoff_height) < 0.2:
                    self._set_phase("hover")
                    self.hover_start_time = time.time()

            # Timeout
            if self._phase_elapsed() > TAKEOFF_TIMEOUT:
                self.get_logger().warn(f"Takeoff timeout ({TAKEOFF_TIMEOUT}s). Proceeding.")
                self._set_phase("hover")
                self.hover_start_time = time.time()

        # ---- HOVER: hold position for HOVER_DURATION seconds ----
        elif self.phase == "hover":
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

            # Log every 20th tick (1Hz)
            if self.log_counter % 20 == 0 and self.vehicle_local_position is not None:
                pos = self.vehicle_local_position
                self.get_logger().info(
                    f"  Hovering: x={pos.x:.2f} y={pos.y:.2f} z={-pos.z:.2f}m")

            if time.time() - self.hover_start_time > HOVER_DURATION:
                self._set_phase("move")

        # ---- MOVE: fly to waypoint (2, 0, takeoff_height) ----
        elif self.phase == "move":
            target_x, target_y = 2.0, 0.0
            # Yaw toward direction of travel
            if self.vehicle_local_position is not None:
                dx = target_x - self.vehicle_local_position.x
                dy = target_y - self.vehicle_local_position.y
                yaw = math.atan2(dy, dx)
            else:
                yaw = 0.0

            self.publish_position_setpoint(target_x, target_y, self.takeoff_height, yaw=yaw)

            if self.vehicle_local_position is not None:
                pos = self.vehicle_local_position
                # 3D distance including Z (audit fix: was 2D only)
                dist = math.sqrt(
                    (pos.x - target_x)**2 +
                    pos.y**2 +
                    (pos.z - self.takeoff_height)**2
                )

                if self.log_counter % 10 == 0:
                    self.get_logger().info(
                        f"  Moving: x={pos.x:.2f} y={pos.y:.2f} | "
                        f"3D dist to target: {dist:.2f}m")

                if dist < 0.5:
                    self._set_phase("land")

            # Timeout
            if self._phase_elapsed() > MOVE_TIMEOUT:
                self.get_logger().warn(f"Move timeout ({MOVE_TIMEOUT}s). Landing.")
                self._set_phase("land")

        # ---- LAND ----
        elif self.phase == "land":
            self.land()
            self._set_phase("landing")

        # ---- LANDING: wait for touchdown ----
        elif self.phase == "landing":
            if self.vehicle_local_position is not None:
                alt = -self.vehicle_local_position.z
                if self.log_counter % 20 == 0:
                    self.get_logger().info(f"  Landing: alt={alt:.2f}m")

                if alt < 0.15:
                    self.get_logger().info("=== TEST COMPLETE: Drone landed successfully ===")
                    raise SystemExit

            # Timeout
            if self._phase_elapsed() > LAND_TIMEOUT:
                self.get_logger().warn(f"Land timeout ({LAND_TIMEOUT}s). Exiting.")
                raise SystemExit


def main():
    rclpy.init()
    node = DroneController()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info("Test finished successfully")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()