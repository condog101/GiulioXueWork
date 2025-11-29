from setuptools import setup

package_name = 'markerless_tracking'
pkg2 = 'markerless_tracking/utils'
pkg3 = 'markerless_tracking/ckpt/pcl'
pkg4 = 'markerless_tracking/ckpt/rgb'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name,pkg2,pkg3,pkg4],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Oceane',
    maintainer_email='o.ouillon22@imperial.ac.uk',
    description='The markerless_tracking package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'markerless = markerless_tracking.markerless_tracking_node:main',
        ],
    },
)
