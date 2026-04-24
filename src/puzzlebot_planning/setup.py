from setuptools import find_packages, setup

package_name = 'puzzlebot_planning'

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
    description='A* path planner and reactive obstacle avoidance for the Puzzlebot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'path_planner_node       = puzzlebot_planning.path_planner_node:main',
            'obstacle_avoidance_node = puzzlebot_planning.obstacle_avoidance_node:main',
        ],
    },
)
