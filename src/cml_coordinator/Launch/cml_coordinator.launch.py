from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # CML Coordinator ノード
        Node(
            package='cml_coordinator',
            executable='cml_coordinator_node', # setup.pyのentry_pointsで指定した名前
            name='cml_coordinator'
        ),
        # TTS ノード
        Node(
            package='cml_coordinator',
            executable='tts_node', # setup.pyのentry_pointsで指定した名前
            name='tts_node'
        )
    ])