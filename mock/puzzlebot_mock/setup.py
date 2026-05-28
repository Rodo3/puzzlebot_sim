from setuptools import find_packages, setup
import os

package_name = 'puzzlebot_mock'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            ['launch/mock_test.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rpzda',
    maintainer_email='rpz.dar14@gmail.com',
    description='Mock publishers for testing the web dashboard without the Puzzlebot.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mock_robot = puzzlebot_mock.mock_robot_publisher:main',
            'mock_camera = puzzlebot_mock.mock_camera_publisher:main',
        ],
    },
)
