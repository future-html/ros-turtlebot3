from setuptools import find_packages, setup

package_name = 'turtlecv'

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
    maintainer='future',
    maintainer_email='future@todo.todo',
    description='TurtleBot3 Computer Vision Package with OpenCV and YOLO',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'opencv = turtlecv.opencv:main',
            'pink_detector = turtlecv.pink_detector:main',
            'turtlecv = turtlecv.turtlecv:main',
        ],
    },
)
