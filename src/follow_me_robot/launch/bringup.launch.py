import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    kobuki_launch_dir = get_package_share_directory('kobuki_node')
    kobuki_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(kobuki_launch_dir, 'launch', 'kobuki_node-launch.py'))
    )

    sllidar_launch_dir = get_package_share_directory('sllidar_ros2')
    sllidar_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(sllidar_launch_dir, 'launch', 'view_sllidar_a1_launch.py')),
        launch_arguments={'serial_port': '/dev/ttyUSB1'}.items()
    )

    tf_footprint_to_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_link']
    )

    tf_link_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser'] 
    )

    slam_launch_dir = get_package_share_directory('slam_toolbox')
    slam_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_launch_dir, 'launch', 'online_async_launch.py'))
    )

    return LaunchDescription([
        kobuki_cmd,
        sllidar_cmd,
        tf_footprint_to_link,
        tf_link_to_laser,
        slam_cmd
    ])
