from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # get config file
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('global_map'),
        'config',
        'global_map_builder_default_config.yaml'
    )

    # create node
    agent_node = Node(
        package='global_map',
        executable='global_map_builder_node',
        name='global_map_builder_node',
        parameters=[config],
        #prefix=['xterm -fa default -fs 10 -xrm "XTerm*selectToClipboard: true" -e gdb --args'],#-ex run 
        output='screen',
        emulate_tty=True
    )

    ld.add_action(agent_node)
    return ld

#break global_map_builder.cpp:20