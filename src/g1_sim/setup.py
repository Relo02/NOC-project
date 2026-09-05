import glob
from setuptools import setup

package_name = 'g1_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob.glob('rviz/*.rviz')),
        ('share/' + package_name + '/assets/g1', glob.glob('assets/g1/*.xml')),
        ('share/' + package_name + '/assets/g1/meshes', glob.glob('assets/g1/meshes/*.STL')),
        ('share/' + package_name + '/description', glob.glob('description/*.urdf')),
        ('share/' + package_name + '/description/meshes', glob.glob('description/meshes/*.STL')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Francesco Pedrini',
    maintainer_email='franci.pedrini@gmail.com',
    description='Simulatore MuJoCo del G1 come impianto per lo stack A*+MPC.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_sim = g1_sim.mujoco_sim:main',
            'key_teleop = g1_sim.key_teleop:main',
            'cloud_self_filter = g1_sim.cloud_self_filter:main',
        ],
    },
)
