from setuptools import find_packages, setup

package_name = 'puzzlebot_logo_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', [
            'models/best.pt',
            'models/best.onnx',
            'models/data.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jorge Reyes',
    maintainer_email='rpz.dar14@gmail.com',
    description='YOLO11n logo detection node for the Puzzlebot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'yolo_node          = puzzlebot_logo_detector.yolo_node:main',
            'record_logos       = puzzlebot_logo_detector.record_logos:main',
        ],
    },
)
