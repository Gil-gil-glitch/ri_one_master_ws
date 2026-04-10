from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        Node(
            package='nlp_system',
            executable='asr_node',
            name='asr_node',
            output='screen'
        ),

        Node(
            package='nlp_system',
            executable='nlp_node',
            name='nlp_node',
            output='screen'
        ),

        Node(
            package='nlp_system',
            executable='dialogue_manager',
            name='dialogue_manager',
            output='screen'
        ),

        # Node(
        #     package='nlp_system',
        #     executable='temp_temp',
        #     name='temp_temp',
        #     output='screen'
        # ),

        Node(
            package='nlp_system',
            executable='task_planner_node',
            name='task_planner_node',
            output='screen'
        ),

        ])
