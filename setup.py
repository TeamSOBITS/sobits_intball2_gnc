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
    ],
    install_requires=['setuptools'],
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
            'fan_control = sobits_intball2_gnc.control.fan_control:main',
        ],
    },
)
