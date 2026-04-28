from setuptools import find_packages, setup

package_name = 'puzzlebot_voice_commands'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rpzda',
    maintainer_email='rpz.dar14@gmail.com',
    description=(
        'Offline voice command recognition using MFCC features and '
        'from-scratch ML models. Phase 1: package structure only.'
    ),
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'prepare_voice_dataset = '
                'puzzlebot_voice_commands.scripts.prepare_dataset:main',
            'train_voice_models = '
                'puzzlebot_voice_commands.scripts.train_models:main',
            'evaluate_voice_models = '
                'puzzlebot_voice_commands.scripts.evaluate_models:main',
            'predict_voice_file = '
                'puzzlebot_voice_commands.scripts.predict_file:main',
        ],
    },
)
