"""
ROS 2 Launch File for Person Learning System
=============================================
Starts the 3-node perception pipeline + mock NLP node for testing.

Architecture:
  /vision_node -> /person_detection -> /person_tracker -> /tracked_person -> /task_planner
                                                                                   ^
  (NLP) /person_profile -----------------------------------------------------------| 
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the Person Learning System."""
    
    return LaunchDescription([
        # === CV Pipeline ===
        
        # Node 1: Vision Node (Person Detection)
        # Publishes: /person_detection
        Node(
            package='person_perception',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[{
                'model_path': 'yolov8n.pt',
                'conf_threshold': 0.25,
                'publish_rate': 30.0,
                'show_debug_window': True,
                'use_webcam': True,       # Set True for webcam, False for RealSense
                'camera_id': 0,
            }]
        ),
        
        # Node 2: Person Tracker (Identity Assignment)
        # Subscribes: /person_detection
        # Publishes:  /tracked_person
        Node(
            package='person_perception',
            executable='person_tracker',
            name='person_tracker',
            output='screen',
            parameters=[{
                'show_debug_window': True,
                'embeddings_path': '',
            }]
        ),
        
        # Node 3: Task Planner (Coordinator)
        # Subscribes: /tracked_person, /person_profile
        # Publishes:  /task_planner/actions
        Node(
            package='person_perception',
            executable='task_planner',
            name='task_planner',
            output='screen',
            parameters=[{
                'db_path': 'person_database.json',
                'show_log': True,
            }]
        ),
        
        # === Testing / Debug ===
        
        # Mock NLP Node (for debugging CV pipeline)
        Node(
            package='person_perception',
            executable='mock_nlp_node',
            name='mock_nlp_node',
            output='screen',
        ),
    ])
