import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 各パッケージのディレクトリを取得
    # 自パッケージ（receptionist_system）
    pkg_receptionist = get_package_share_directory('receptionist_system')
    
    # 他の依存パッケージ
    pkg_follow_me = get_package_share_directory('follow_me_robot')
    pkg_realsense = get_package_share_directory('realsense2_camera')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')
    pkg_omx_bringup = get_package_share_directory('open_manipulator_x_bringup')

    # 各種設定
    map_path = os.path.join(os.path.expanduser('~'), 'ri_one_master_ws', 'my_map.yaml')

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),

        # 1. Robot Bringup (Kobuki等の足回り)
        IncludeLaunchDescription(
            # follow_me_robot の bringup を使用
            PythonLaunchDescriptionSource(os.path.join(pkg_follow_me, 'launch', 'bringup.launch.py'))
        ),

        # 2. RealSense
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_realsense, 'launch', 'rs_launch.py')),
            launch_arguments={'align_depth.enable': 'true'}.items()
        ),

        # 3. Cartographer
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_follow_me, 'launch', 'cartographer.launch.py'))
        ),

        # 4. Navigation2 (修正済みの navigation.launch.py を呼び出す)
        IncludeLaunchDescription(
            # 先ほど修正した follow_me_robot 側の navigation.launch.py を参照
            PythonLaunchDescriptionSource(os.path.join(pkg_follow_me, 'launch', 'navigation.launch.py')),
            launch_arguments={
                'use_sim_time': 'false',
                'autostart': 'true',
                'map': map_path
            }.items()
        ),

        # 5. Open Manipulator-X
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_omx_bringup, 'launch', 'hardware.launch.py')),
            launch_arguments={'port_name': '/dev/ttyACM0'}.items()
        ),

        # 6. Receptionist System (本丸)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_receptionist, 'launch', 'receptionist.launch.py'))
        ),
    ])