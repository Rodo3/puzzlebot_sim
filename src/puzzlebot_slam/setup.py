from setuptools import find_packages, setup

package_name = 'puzzlebot_slam'

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
    maintainer='Jesus Martinez',
    maintainer_email='chat4Claude@outlook.com',
    description='Monte Carlo Localization SLAM node for the Puzzlebot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'slam_node = puzzlebot_slam.slam_node:main',
        ],
    },
)
