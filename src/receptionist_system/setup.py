from setuptools import setup, find_packages # find_packagesを追加
import os
from glob import glob

package_name = 'receptionist_system'

setup(
    name=package_name,
    version='0.0.0',
    # 修正ポイント1: find_packages() により、core などのサブディレクトリを自動でパッケージとして認識させます
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 修正ポイント2: 打ち間違いを防ぐため os.path.join を使用（既存のままでも可）
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ri-one',
    maintainer_email='youkongmo@gmail.com',
    description='Receptionist system for RoboCup@Home',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'asr_node = receptionist_system.asr_node:main',
            'nlp_node = receptionist_system.nlp_node:main',
            'tts_node = receptionist_system.tts_node:main',
            'vision_node = receptionist_system.vision_node:main',
            'dialogue_manager = receptionist_system.dialogue_manager:main',
            'task_planner = receptionist_system.task_planner_node:main',
        ],
    },
)