from setuptools import find_packages, setup

package_name = 'homework_01_transforms'

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
    maintainer='Mario Martinez',
    maintainer_email='mario.mtz@manchester-robotics.com',
    description='Homework 1 - TF transforms and kinematic simulation of Puzzlebot circular trajectory',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'joint_state_publisher = homework_01_transforms.joint_state_publisher:main',
        ],
    },
)
