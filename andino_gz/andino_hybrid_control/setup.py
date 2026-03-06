from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'andino_hybrid_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gowtham Ammanamanchi',
    maintainer_email='gowtham@inspection.robot',
    description='Hybrid Nonlinear Control + SAC RL Disturbance Compensation for Andino robot',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'trajectory_generator = andino_hybrid_control.trajectory_generator_node:main',
            'nonlinear_controller = andino_hybrid_control.nonlinear_controller_node:main',
            'rl_compensator = andino_hybrid_control.rl_disturbance_compensator_node:main',
            'hybrid_controller = andino_hybrid_control.hybrid_controller_node:main',
            'performance_monitor = andino_hybrid_control.performance_monitor_node:main',
            'data_logger = andino_hybrid_control.data_logger_node:main',
        ],
    },
)
