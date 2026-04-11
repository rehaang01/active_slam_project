import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

"""
full_simulation_surrogate.launch.py — Launch file WITHOUT RTAB-Map

Launches:
  1. MicroXRCEAgent (PX4 ↔ ROS2 bridge)
  2. ROS-GZ bridges for camera_info, LiDAR, camera, depth_camera
  3. Static TF publishers for camera + LiDAR frames
  
Does NOT launch:
  - RTAB-Map (replaced by slam_surrogate.py running inside the env)
  - keyboard-mavsdk-test.py (RL agent controls the drone)
  
PX4 SITL + Gazebo must be started separately:
  cd /root/PX4-Autopilot
  PX4_GZ_WORLD=warehouse PX4_SYS_AUTOSTART=4002 \
  PX4_GZ_MODEL_POSE="0.0,0.0,1.0,0.00,0,3.14" \
  PX4_GZ_MODEL=x500_depth ./build/px4_sitl_default/bin/px4

Usage:
  ros2 launch simulation_launch full_simulation_surrogate.launch.py
  
Or copy this file over the original:
  cp full_simulation_surrogate.launch.py /path/to/full_simulation.launch.py
"""


def generate_launch_description():

    # MicroXRCEAgent — bridges PX4 DDS ↔ ROS2
    micro_xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen'
    )

    # ROS-GZ bridges — camera info + LiDAR
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
            '/world/warehouse/model/x500_depth_0/link/link/sensor/lidar_2d_v2/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
        ],
        remappings=[
            ('/world/warehouse/model/x500_depth_0/link/link/sensor/lidar_2d_v2/scan', '/scan'),
        ],
        output='screen'
    )

    # Image bridges — RGB camera + depth camera
    image_bridge_camera = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera'],
        output='screen'
    )

    image_bridge_depth = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/depth_camera'],
        output='screen'
    )

    # Static TF publishers — camera frames
    static_tf_imx214 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'x500_depth_0/OakD-Lite/base_link',
            '--child-frame-id', 'x500_depth_0/OakD-Lite/base_link/IMX214',
        ],
        output='screen'
    )

    static_tf_stereo = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'x500_depth_0/OakD-Lite/base_link',
            '--child-frame-id', 'x500_depth_0/OakD-Lite/base_link/StereoOV7251',
        ],
        output='screen'
    )

    # Static TF — LiDAR frame
    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.12', '--y', '0', '--z', '0.26',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'x500_depth_0/base_link',
            '--child-frame-id', 'x500_depth_0/link/lidar_2d_v2',
        ],
        output='screen'
    )

    # NOTE: No RTAB-Map launch — slam_surrogate.py handles mapping
    return LaunchDescription([
        micro_xrce_agent,
        parameter_bridge,
        image_bridge_camera,
        image_bridge_depth,
        static_tf_imx214,
        static_tf_stereo,
        static_tf_lidar,
    ])