#!/bin/bash
# ============================================================
# add_lidar_to_drone.sh
# Run this INSIDE the Docker container (rtabmap_ros2)
#
# Adds a 360-degree 2D LiDAR sensor to the x500_depth drone
# model for obstacle detection and reactive safety.
#
# Usage:
#   sudo docker exec -it rtabmap_ros2 bash
#   bash /root/active_slam_project/add_lidar_to_drone.sh
# ============================================================

set -e  # Exit on error

echo "============================================"
echo " Adding 2D LiDAR to x500_depth drone model"
echo "============================================"

# Step 1: Find the model SDF file
MODEL_DIR="/root/PX4-Autopilot/Tools/simulation/gz/models/x500_depth"
SDF_FILE="${MODEL_DIR}/model.sdf"

if [ ! -f "$SDF_FILE" ]; then
    echo "ERROR: Model SDF not found at: $SDF_FILE"
    echo "Searching for x500_depth model..."
    find /root/PX4-Autopilot -name "model.sdf" -path "*x500_depth*" 2>/dev/null
    echo ""
    echo "If found at a different path, edit MODEL_DIR in this script."
    exit 1
fi

echo "Found model SDF: $SDF_FILE"

# Step 2: Check if LiDAR already added
if grep -q "lidar_link" "$SDF_FILE"; then
    echo "LiDAR sensor already present in the model. Skipping."
    exit 0
fi

# Step 3: Create backup
BACKUP="${SDF_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$SDF_FILE" "$BACKUP"
echo "Backup created: $BACKUP"

# Step 4: Find the insertion point
# We need to insert the LiDAR link + joint just before the closing </model> tag.
# The x500_depth model has a top-level <model> that we need to add to.
echo "Injecting LiDAR sensor XML..."

# The LiDAR XML block to inject
LIDAR_XML='
    <!-- ================================================== -->
    <!-- 2D LiDAR sensor for 360-degree obstacle detection  -->
    <!-- Added by Active SLAM project                       -->
    <!-- ================================================== -->
    <link name="lidar_link">
      <pose relative_to="base_link">0 0 -0.05 0 0 0</pose>
      <inertial>
        <mass>0.015</mass>
        <inertia>
          <ixx>1e-5</ixx>
          <iyy>1e-5</iyy>
          <izz>1e-5</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <sensor name="lidar_sensor" type="gpu_lidar">
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <topic>lidar</topic>
        <lidar>
          <scan>
            <horizontal>
              <samples>360</samples>
              <resolution>1</resolution>
              <min_angle>-3.14159265</min_angle>
              <max_angle>3.14159265</max_angle>
            </horizontal>
            <vertical>
              <samples>1</samples>
              <resolution>1</resolution>
              <min_angle>0</min_angle>
              <max_angle>0</max_angle>
            </vertical>
          </scan>
          <range>
            <min>0.1</min>
            <max>10.0</max>
            <resolution>0.01</resolution>
          </range>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </lidar>
        <visualize>false</visualize>
      </sensor>
    </link>
    <joint name="lidar_joint" type="fixed">
      <parent>base_link</parent>
      <child>lidar_link</child>
    </joint>
'

# Step 5: Insert before the LAST closing </model> tag
# Use Python for reliable XML manipulation (sed can break on multi-line)
python3 << PYEOF
import re

with open("$SDF_FILE", "r") as f:
    content = f.read()

# Find the last </model> tag and insert before it
lidar_xml = '''$LIDAR_XML'''

# Find the position of the last </model>
last_model_close = content.rfind("</model>")
if last_model_close == -1:
    print("ERROR: Could not find </model> tag in SDF file!")
    exit(1)

# Insert LiDAR XML before the last </model>
new_content = content[:last_model_close] + lidar_xml + "\n  " + content[last_model_close:]

with open("$SDF_FILE", "w") as f:
    f.write(new_content)

print("LiDAR XML injected successfully.")
PYEOF

# Step 6: Verify
echo ""
echo "Verifying LiDAR was added..."
if grep -q "lidar_link" "$SDF_FILE" && grep -q "gpu_lidar" "$SDF_FILE"; then
    echo "SUCCESS: LiDAR sensor added to x500_depth model."
else
    echo "ERROR: Verification failed. Restoring backup..."
    cp "$BACKUP" "$SDF_FILE"
    exit 1
fi

# Step 7: Show what to do next
echo ""
echo "============================================"
echo " NEXT STEPS"
echo "============================================"
echo ""
echo "1. The launch file also needs updating to bridge the"
echo "   LiDAR Gazebo topic to ROS2."
echo ""
echo "2. Replace the launch file:"
echo "   cp /root/active_slam_project/launch/full_simulation.launch.py \\"
echo "      \$(ros2 pkg prefix simulation_launch)/share/simulation_launch/launch/"
echo ""
echo "3. Restart the simulation:"
echo "   tmuxinator stop rtabmap_ros2_gazebo"
echo "   tmuxinator rtabmap_ros2_gazebo"
echo ""
echo "4. Verify the LiDAR topic is publishing:"
echo "   ros2 topic list | grep scan"
echo "   ros2 topic echo /scan --once"
echo ""
echo "If /scan does not appear, check the Gazebo topic:"
echo "   gz topic -l | grep lidar"
echo "Then update the bridge topic name in the launch file."
echo ""