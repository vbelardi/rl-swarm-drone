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
        ],
    },
    data_files=[
        # This installs the package.xml for ROS2 indexing.
        ('share/ament_index/resource_index/packagenames', ['package.xml']),
        # This copies your launch directory and its files.
        ('share/rl_interface/launch', ['launch/rl.launch.py']),
    ],
)
