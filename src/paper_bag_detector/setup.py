from setuptools import setup

package_name = 'paper_bag_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Paper bag detection and distance estimation',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'detector_node = paper_bag_detector.detector_node:main',
            'distance_node = paper_bag_detector.distance_node:main',
        ],
    },
)