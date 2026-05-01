from setuptools import find_packages, setup

package_name = 'open_challenge_2026'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ri-one',
    maintainer_email='youkongmo@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'home_task_coordinator = open_challenge_2026.home_task_coordinator:main',
            'home_task_follow_me_node = open_challenge_2026.home_task_follow_me_node:main',
            'home_task_approach_node = open_challenge_2026.home_task_approaching_person_node:main',
            'home_task_gesture_node = open_challenge_2026.home_task_gesture_identification_node:main',
            'home_task_targetting_node = open_challenge_2026.home_task_person_targetting_node:main',
            'home_task_tts_node = open_challenge_2026.home_task_tts_node:main',
        ],
    },
)
