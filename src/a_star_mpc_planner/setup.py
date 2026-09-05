import glob
from setuptools import setup

package_name = 'a_star_mpc_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorenzo Ortolani',
    maintainer_email='lorenzo@example.com',
    description='A* path planner with MPC trajectory tracking for Unitree G1 humanoid navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'a_star_node = a_star_mpc_planner.a_star_node:main',
            'mpc_node = a_star_mpc_planner.mpc_node:main',
            'odom_to_pose_node = a_star_mpc_planner.odom_to_pose_node:main',
            'setpoint_to_cmd_vel_node = a_star_mpc_planner.setpoint_to_cmd_vel_node:main',
            'nav_graph_node = a_star_mpc_planner.navigation_graph_node:main',
            'pose_from_tf = a_star_mpc_planner.pose_from_tf:main',
        ],
    },
)
