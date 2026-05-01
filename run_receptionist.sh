#!/bin/bash

# Configuration
SESSION="reception_sys"
WORKSPACE_DIR="$HOME/ri_one_master_ws"

echo "Starting Receptionist System in tmux..."

# Ensure we are in the workspace
cd "$WORKSPACE_DIR" || exit

# Check if tmux is installed
if ! command -v tmux &> /dev/null
then
    echo "tmux is not installed. Please install it using: sudo apt-get install tmux"
    exit 1
fi

# Destroy session if it already exists safely
tmux kill-session -t $SESSION 2>/dev/null

# Create a new session in detached mode (starting at window 0)
tmux new-session -d -s $SESSION -n "bringup"

# --- Window 0: Bringup (Robot Base & Lidar) ---
tmux send-keys -t $SESSION:0 "source install/setup.bash && ros2 launch follow_me_robot bringup.launch.py" C-m

# --- Window 1: RealSense Camera ---
tmux new-window -t $SESSION:1 -n "realsense"
tmux send-keys -t $SESSION:1 "source install/setup.bash && ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true" C-m

# --- Window 2: SLAM (Cartographer) ---
tmux new-window -t $SESSION:2 -n "cartographer"
tmux send-keys -t $SESSION:2 "source install/setup.bash && ros2 launch follow_me_robot cartographer.launch.py" C-m

# --- Window 3: Navigation ---
tmux new-window -t $SESSION:3 -n "navigation"
tmux send-keys -t $SESSION:3 "source install/setup.bash && ros2 launch follow_me_robot navigation.launch.py use_sim_time:=false autostart:=true map:=$HOME/ri_one_master_ws/my_map.yaml" C-m

# --- Window 4: RViz ---
tmux new-window -t $SESSION:4 -n "rviz"
tmux send-keys -t $SESSION:4 "source install/setup.bash && ros2 run rviz2 rviz2 -d \$(ros2 pkg prefix nav2_bringup)/share/nav2_bringup/rviz/nav2_default_view.rviz" C-m

# --- Window 5: Manipulator Arm ---
tmux new-window -t $SESSION:5 -n "arm"
tmux send-keys -t $SESSION:5 "source install/setup.bash && ros2 launch open_manipulator_x_bringup hardware.launch.py port_name:=/dev/ttyACM0" C-m

# --- Window 6: Receptionist System ---
tmux new-window -t $SESSION:6 -n "ai_logic"
tmux send-keys -t $SESSION:6 "source install/setup.bash && ros2 launch receptionist_system receptionist.launch.py" C-m


echo "=========================================================================="
echo "All nodes are now launching in the background using tmux!"
echo ""
echo "To view the logs and manage the terminals, run:"
echo "  tmux attach-session -t $SESSION"
echo ""
echo "Tmux Quick Cheatsheet (while attached):"
echo "  • Switch between windows: press [Ctrl + b] and then a number [0 - 6]"
echo "  • Go to Next window:      press [Ctrl + b] and then [n]"
echo "  • Go to Prev window:      press [Ctrl + b] and then [p]"
echo "  • Detach (exit without closing): press [Ctrl + b] and then [d]"
echo "  • Close everything:       tmux kill-session -t $SESSION"
echo "=========================================================================="
