from setuptools import setup

setup(
    name="rl_interface",
    version="0.0.1",
    packages=["rl_interface"],
    install_requires=[
        "gymnasium",
    ],
    entry_points={
        "console_scripts": [
            "rl_node = rl_interface.rl_node:main",
            "grid_world = rl_interface.grid_world:main",
            "2_drones = rl_interface.2_drones:main",
            "random_exploration = rl_interface.random_exploration:main",
            "3drones_test_no_coord = rl_interface.3drones_test_no_coord:main",
        ],
    },
    data_files=[
        # This installs the package.xml for ROS2 indexing.
        ('share/ament_index/resource_index/packagenames', ['package.xml']),
        # This copies your launch directory and its files.
        ('share/rl_interface/launch', ['launch/rl.launch.py']),
        ('share/rl_interface/launch', ['launch/random_explo.launch.py']),
    ],
)
