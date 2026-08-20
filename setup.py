from glob import glob
from setuptools import find_packages, setup

package_name = 'sobits_intball2_gnc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/maps', glob('maps/*.yaml')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='rg-msi-01',
    maintainer_email='rg-msi-01@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'location_broadcaster = sobits_intball2_gnc.navigation.location_broadcaster:main',
            'location_setting = sobits_intball2_gnc.navigation.location_setting:main',
            # Control-system orchestrator (the single control node).
            'control = sobits_intball2_gnc.control.control:main',
            # ros/ wrapper manual-test entry points.
            'fan_duty_publisher = sobits_intball2_gnc.control.ros.fan_duty_publisher:main',
            'imu_subscriber = sobits_intball2_gnc.control.ros.imu_subscriber:main',
            'tf_client = sobits_intball2_gnc.common.ros.tf_client:main',
            'pose_array_subscriber = sobits_intball2_gnc.control.ros.pose_array_subscriber:main',
            'multi_dof_joint_trajectory_subscriber = '
            'sobits_intball2_gnc.control.ros.multi_dof_joint_trajectory_subscriber:main',
            'path_publisher = sobits_intball2_gnc.guidance.ros.path_publisher:main',
            'multi_dof_joint_trajectory_publisher = '
            'sobits_intball2_gnc.guidance.ros.multi_dof_joint_trajectory_publisher:main',
            'ctl_command_action_server = '
            'sobits_intball2_gnc.guidance.ros.ctl_command_action_server:main',
        ],
    },
)
