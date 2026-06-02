from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install the maze PNG map alongside the package Python files
        (os.path.join('share', package_name, 'puzzlebot_slam'),
            glob(os.path.join('puzzlebot_slam', '*.png')) +
            glob(os.path.join('puzzlebot_slam', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jesus Martinez',
    maintainer_email='chat4Claude@outlook.com',
    description='Monte Carlo Localization SLAM node for the Puzzlebot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'slam_node          = puzzlebot_slam.slam_node:main',
            'mcl                = puzzlebot_slam.mcl:main',
            'map_server_node    = puzzlebot_slam.map_server_node:main',
        ],
    },
)
