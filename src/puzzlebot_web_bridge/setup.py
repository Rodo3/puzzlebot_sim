from setuptools import find_packages, setup

package_name = 'puzzlebot_web_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'fastapi',
        'uvicorn[standard]',
        'websockets',
    ],
    zip_safe=True,
    maintainer='Puzzlebot Dev',
    maintainer_email='rpz.dar14@gmail.com',
    description='WebSocket bridge between ROS 2 topics and the Puzzlebot web dashboard.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bridge_node = puzzlebot_web_bridge.bridge_node:main',
        ],
    },
)
