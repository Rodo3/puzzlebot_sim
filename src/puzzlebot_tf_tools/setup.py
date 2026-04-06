from setuptools import find_packages, setup

package_name = 'puzzlebot_tf_tools'

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
    description='Reusable TF utilities and transformation helpers for Puzzlebot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [],
    },
)
