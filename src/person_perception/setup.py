"""
ROS 2 Setup for Person Perception Package
"""
from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'person_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',
        'ultralytics>=8.0.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'pyrealsense2>=2.54.1',
        'insightface>=0.7.3',
        'onnxruntime>=1.15.0',
        'scipy>=1.10.0',
    ],
    zip_safe=True,
    maintainer='Jonathan Setiawan',
    maintainer_email='jonathan@example.com',
    description='Research-Grade Person Perception System for RoboCup JapanOpen',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Legacy monolithic node
            'person_node = person_perception.nodes.person_node:main',
            # New 3-node architecture
            'vision_node = person_perception.nodes.vision_node:main',
            'person_tracker = person_perception.nodes.person_tracker_node:main',
            'task_planner = person_perception.nodes.task_planner_node:main',
            # Tools
            'mock_nlp_node = tools.mock_nlp_node:main',
        ],
    },
)
