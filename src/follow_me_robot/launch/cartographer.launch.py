import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_dir = get_package_share_directory('follow_me_robot')
    
    return LaunchDescription([
        # 1. 静的TFの発行 (追加箇所)
        # base_footprint から見て laser がどこにあるかを定義します
        # 引数: x y z yaw pitch roll parent_frame child_frame
        # 下記は「ロボットの中心から前方0cm、高さ10cm」に設置されている例です
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_laser',
            arguments=['0.0', '0.0', '0.1', '0.0', '0.0', '0.0', 'base_footprint', 'laser'],
        ),

        # 2. Cartographer本体の起動
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[
                '-configuration_directory', os.path.join(package_dir, 'config'),
                '-configuration_basename', 'my_cartographer.lua'
            ],
        ),

        # 3. 地図をROS標準のOccupancyGrid形式に変換するノード
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='occupancy_grid_node',
            parameters=[{'resolution': 0.05}],
        ),
    ])