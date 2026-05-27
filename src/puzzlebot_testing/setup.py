from setuptools import find_packages, setup

package_name = 'puzzlebot_testing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Script wrapper para ros2 run puzzlebot_testing odometry_validator
        ('lib/' + package_name, ['scripts/odometry_validator']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jesus Martinez',
    maintainer_email='chat4Claude@outlook.com',
    description='Herramientas de diagnóstico y validación del Puzzlebot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'odometry_validator = puzzlebot_testing.odometry_validator:main',
        ],
    },
)
