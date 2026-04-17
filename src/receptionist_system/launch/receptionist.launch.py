import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. ASR Node (音声認識)
        Node(
            package='receptionist_system',
            executable='asr_node',
            name='asr_node',
            parameters=[{
                'silence_threshold': 1500.0, # あなたの環境に合わせた数値
                'whisper_model': 'small',
                'use_gpu': True
            }],
            output='screen'
        ),

        # 2. NLP Node (自然言語処理)
        Node(
            package='receptionist_system',
            executable='nlp_node',
            name='nlp_node',
            output='screen'
        ),

        # 3. Task Planner Node (司令塔)
        Node(
            package='receptionist_system',
            executable='task_planner',
            name='task_planner',
            output='screen'
        ),

        # 4. Dialogue Manager (会話管理)
        Node(
            package='receptionist_system',
            executable='dialogue_manager',
            name='dialogue_manager',
            output='screen'
        ),

        # 5. TTS Node (オフライン音声合成)
        Node(
            package='receptionist_system',
            executable='tts_node',
            name='tts_node',
            output='screen'
        ),

        # 6. Vision Node (画像認識)
        Node(
            package='receptionist_system',
            executable='vision_node',
            name='vision_node',
            output='screen'
        ),
    ])