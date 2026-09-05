import glob
from setuptools import setup

package_name = 'robot_real_goal_manager'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorenzo Ortolani',
    maintainer_email='lorenzo.ortolani@talosrobotics.ai',
    description='Goal-input frontend (RViz relay + waypoint mission runner) for the A*+MPC stack.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goal_relay_node      = robot_real_goal_manager.goal_relay_node:main',
            'mission_runner_node  = robot_real_goal_manager.mission_runner_node:main',
        ],
    },
)
