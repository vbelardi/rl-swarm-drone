from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_share_dir = get_package_share_directory('rl_interface')

    config = os.path.join(package_share_dir, 'config', 'rl_config.yaml')

    rl_node = Node(
        package='rl_interface',     
        executable='random_exploration',         
        name='random_exploration',              
        parameters=[config],           
        output='screen',                
        emulate_tty=True                 
    )

    ld = LaunchDescription()
    ld.add_action(rl_node)
    return ld
