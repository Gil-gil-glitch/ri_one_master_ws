from setuptools import find_packages, setup

package_name = 'paper_bag_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
                      'setuptools',
                      'opencv-python',
                      'inference_sdk',
                     ],
    zip_safe=True,
    maintainer='ri-one',
    maintainer_email='ri-one@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'real_to_bag = paper_bag_detection.real_to_bag:main',
            'paper_bag = paper_bag_detection.paper_bag:main',

            #ros2 run paper_bag_detection position_pub
            # 'position_pub = paper_bag_detection.pub_sub_node:main',
            # 'bag_pub = paper_bag_detection.bag_pub_node:main',
        ],
    },
)
