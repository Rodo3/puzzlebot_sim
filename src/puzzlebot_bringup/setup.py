from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mario Martinez',
    maintainer_email='mario.mtz@manchester-robotics.com',
    description='Launch files for the Puzzlebot simulation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mock_encoders = puzzlebot_bringup.mock_encoders:main',
            'smoke_test    = puzzlebot_bringup.smoke_test:main',
        ],
    },
)
