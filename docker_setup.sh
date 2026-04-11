#!/bin/bash
# ============================================================
# docker_setup.sh — One-time setup for Active SLAM simulation
#
# Run this ONCE after creating a fresh Docker container.
# Not needed after docker stop/start (changes persist).
# Only needed after docker rm + docker run (fresh container).
#
# Usage:
#   sudo docker exec -it rtabmap_ros2 bash
#   bash /root/active_slam_project/docker_setup.sh
# ============================================================

set -e
echo "============================================"
echo " Active SLAM — Docker Setup"
echo "============================================"

# --- 1. Add LiDAR to x500_depth drone model ---
MODEL_SDF="/root/PX4-Autopilot/Tools/simulation/gz/models/x500_depth/model.sdf"

if grep -q "lidar_2d_v2" "$MODEL_SDF" 2>/dev/null; then
    echo "[1/3] LiDAR already in drone model. Skipping."
else
    echo "[1/3] Adding LiDAR to x500_depth model..."
    cp "$MODEL_SDF" "${MODEL_SDF}.backup"
    cp /root/active_slam_project/model.sdf "$MODEL_SDF"
    echo "  Done. Backup at ${MODEL_SDF}.backup"
fi

# --- 2. Update launch file with LiDAR bridge ---
LAUNCH_DIR=$(ros2 pkg prefix simulation_launch 2>/dev/null)/share/simulation_launch/launch
if [ -d "$LAUNCH_DIR" ]; then
    echo "[2/3] Updating launch file..."
    cp /root/active_slam_project/launch/full_simulation.launch.py "$LAUNCH_DIR/"
    echo "  Done. Updated at $LAUNCH_DIR"
else
    echo "[2/3] WARNING: simulation_launch package not found."
    echo "  Run: source /opt/ros/humble/setup.bash"
    echo "  Then re-run this script."
fi

# --- 3. Install Python dependencies ---
echo "[3/3] Installing Python dependencies..."
pip3 install protobuf==3.20.1
pip3 install stable-baselines3[extra] sb3-contrib gymnasium numpy==1.24.4

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " To start simulation:"
echo "   tmuxinator rtabmap_ros2_gazebo"
echo ""
echo " To run pipeline test:"
echo "   cd /root/active_slam_project"
echo "   python3 test_pipeline.py"
echo "============================================"   