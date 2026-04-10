# Person Perception System

This package contains the ROS 2 person learning and recognition system. The complete pipeline performs person detection (using YOLOv8), extracts and matches facial embeddings (using InsightFace) to recognize specific individuals, handles identity tracking, and performs task planning based on the detected person profiles.

## Prerequisites
- The workspace requires a Python virtual environment (`venv`) with dependencies installed from `requirements.txt` in the workspace root (which includes `insightface` and `onnxruntime-gpu` for facial recognition).
- The YOLOv8 model (`yolov8n.pt`) and the `person_database.json` should be available in the workspace root.

## Face Registration

Before the system can recognize you, you must register your face so that the system can learn your facial embeddings.

To register a new user face, run the provided registration script:
```bash
cd ~/ri_one_master_ws
source venv/bin/activate
python3 src/person_perception/tools/register_face.py
```
1. Position your face in front of the webcam.
2. Press `S` to capture and process your face.
3. Provide your name in the terminal.
4. Your facial embedding will be generated as a `.npy` file and saved inside `src/person_perception/data/faces/`.

## Setup Instructions

1. **Activate the Virtual Environment**
   Before building or running the nodes, ensure the virtual environment is activated:
   ```bash
   cd ~/ri_one_master_ws
   source venv/bin/activate
   ```

2. **Build the Workspace**
   Build the package using `colcon`:
   ```bash
   colcon build --packages-select person_perception
   ```

3. **Source the Workspace**
   Source the built workspace setup file:
   ```bash
   source install/setup.bash
   ```

## Running the System

You can run the entire perception system using the provided launch file:

```bash
ros2 launch person_perception perception_system.launch.py
```

This launch file starts the following components:
- **`vision_node`**: Runs YOLOv8 person detection and manages perception data. (Check the launch file parameters to switch between a webcam and a RealSense camera).
- **`person_tracker`**: Handles identity tracking. It compares real-time facial embeddings with the `.npy` files created during Face Registration to assign matching identities.
- **`task_planner`**: Coordinates tasks based on the recognized persons and logs interactions in the root `person_database.json` file.
- **`mock_nlp_node`**: Included for testing and debugging interaction.

## Files & Directories
- `src/person_perception/data/faces/`: Directory containing all `.npy` facial embeddings files computed from registered faces.
- `src/person_perception/tools/register_face.py`: Enrollment utility to add new faces.
- `person_database.json` (Workspace root): Stores recognized user profiles, interactions, attributes, and tracking metadata.
- `yolov8n.pt` (Workspace root): The YOLOv8 model used for detection.
