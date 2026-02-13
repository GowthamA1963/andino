#!/usr/bin/env python3
# Copyright 2024 Ekumen, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Launch file for Andino Gazebo simulation with Nav2 navigation enabled.

This launch file automatically configures and starts all components required for Nav2:
- Gazebo simulation
- Robot spawning
- ROS-Gazebo bridge
- Nav2 stack (localization, planning, control)
- RViz with Nav2 configuration
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Andino Nav2 simulation."""
    pkg_andino_gz = get_package_share_directory('andino_gz')

    # Declare launch arguments
    world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='depot.sdf',
        description='Name of the world to load. Should match with map for proper Nav2 operation.'
    )
    
    map_name_arg = DeclareLaunchArgument(
        'map',
        default_value='depot',
        description='Name of the map to load for Nav2. Should match the world_name.'
    )
    
    robots_arg = DeclareLaunchArgument(
        'robots',
        default_value="andino={x: 0., y: 0., z: 0.1, yaw: 0.};",
        description='Robots to spawn. Multiple robots can be specified separated by semicolon.'
    )
    
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([pkg_andino_gz, 'config', 'nav2_params.yaml']),
        description='Full path to the Nav2 parameters file.'
    )
    
    gui_config_arg = DeclareLaunchArgument(
        'gui_config',
        default_value='default.config',
        description='Name of the Gazebo GUI configuration file to load.'
    )
    
    autostart_arg = DeclareLaunchArgument(
        'autostart',
        default_value='True',
        choices=['True', 'False'],
        description='If True, the simulation starts automatically.'
    )
    
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='True',
        choices=['True', 'False'],
        description='If True, launch RViz with Nav2 configuration.'
    )
    
    ros_bridge_arg = DeclareLaunchArgument(
        'ros_bridge',
        default_value='True',
        choices=['True', 'False'],
        description='If True, run ROS-Gazebo bridge nodes.'
    )

    # Get launch configurations
    world_name = LaunchConfiguration('world_name')
    map_name = LaunchConfiguration('map')
    robots = LaunchConfiguration('robots')
    params_file = LaunchConfiguration('params_file')
    gui_config = LaunchConfiguration('gui_config')
    autostart = LaunchConfiguration('autostart')
    rviz = LaunchConfiguration('rviz')
    ros_bridge = LaunchConfiguration('ros_bridge')

    # Odom to TF publisher node
    # This node publishes the odom -> base_link transform from the /odom topic
    # Workaround for Gazebo OdometryPublisher not reliably publishing TF
    odom_to_tf_node = Node(
        package='andino_gz',
        executable='odom_to_tf_publisher.py',
        name='odom_to_tf_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Include the main andino_gz launch file with Nav2 enabled
    andino_gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_andino_gz, 'launch', 'andino_gz.launch.py')
        ),
        launch_arguments={
            'world_name': world_name,
            'map': map_name,
            'robots': robots,
            'params_file': params_file,
            'gui_config': gui_config,
            'autostart': autostart,
            'rviz': rviz,
            'ros_bridge': ros_bridge,
            'nav2': 'True',  # Enable Nav2
        }.items(),
    )

    # Create and return launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(world_name_arg)
    ld.add_action(map_name_arg)
    ld.add_action(robots_arg)
    ld.add_action(params_file_arg)
    ld.add_action(gui_config_arg)
    ld.add_action(autostart_arg)
    ld.add_action(rviz_arg)
    ld.add_action(ros_bridge_arg)
    
    # Add odom to TF publisher node
    ld.add_action(odom_to_tf_node)
    
    # Add main launch
    ld.add_action(andino_gz_launch)
    
    return ld
