from setuptools import find_packages, setup

package_name = 'puzzlebot_perception'

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
    description='Camera, ArUco, and YOLO detection nodes for the Puzzlebot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'aruco_node         = puzzlebot_perception.aruco_node:main',
            'image_viewer_node  = puzzlebot_perception.image_viewer_node:main',
            'calib_capture_node = puzzlebot_perception.calib_capture_node:main',
            'calib_compute_node = puzzlebot_perception.calib_compute_node:main',
            'calib_apply_node   = puzzlebot_perception.calib_apply_node:main',
            'qr_node            = puzzlebot_perception.qr_node:main',
        ],
    },
)
