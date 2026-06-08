from glob import glob

from setuptools import find_packages, setup

package_name = 'puzzlebot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jesus Martinez',
    maintainer_email='chat4Claude@outlook.com',
    description='High-level mission state machine for the Puzzlebot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mission_manager_node = puzzlebot_control.mission_manager_node:main',
            'fork_publisher_node  = puzzlebot_control.fork_publisher_node:main',
            'fork_mock_node       = puzzlebot_control.fork_mock_node:main',
            'qr_mock_node         = puzzlebot_control.qr_mock_node:main',
            'yolo_mock_node       = puzzlebot_control.yolo_mock_node:main',
            'voice_fsm_router_node = puzzlebot_control.voice_fsm_router_node:main',
        ],
    },
)
