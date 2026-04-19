import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # 1Kobuki Base - Explicitly set to USB1

    kobuki_launch_dir = get_package_share_directory('kobuki_node')
    kobuki_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(kobuki_launch_dir, 'launch', 'kobuki_node-launch.py')),
        launch_arguments={'device_port': '/dev/ttyUSB0'}.items() 
    )

    # 2. LIDAR - Explicitly set to USB0
    sllidar_launch_dir = get_package_share_directory('sllidar_ros2')
    sllidar_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(sllidar_launch_dir, 'launch', 'view_sllidar_a1_launch.py')),
        launch_arguments={
            'serial_port': '/dev/ttyUSB1',
            'serial_baudrate': '115200',
            'frame_id': 'laser',
            'inverted': 'false',
            'angle_compensate': 'true'
        }.items()
    )

    # 3. Static Transforms (Updated to modern ROS 2 style to remove warnings)
    tf_footprint_to_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0', '--frame-id', 'base_footprint', '--child-frame-id', 'base_link']
    )

    tf_link_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0.1', '--yaw', '0', '--pitch', '0', '--roll', '0', '--frame-id', 'base_link', '--child-frame-id', 'laser']
    )

    return LaunchDescription([
        kobuki_cmd,
        sllidar_cmd,
        tf_footprint_to_link,
        tf_link_to_laser
    ])