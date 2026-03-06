#!/usr/bin/env python3
"""
Hybrid Control Launch File
===========================
Launches:
  1. Gazebo simulation (andino_gz) with depot world
  2. Trajectory Generator
  3. Nonlinear Controller (Feedback Linearization)
  4. RL Disturbance Compensator (SAC)
  5. Hybrid Controller (u_model + u_RL → /cmd_vel)
  6. Performance Monitor

Usage:
  ros2 launch andino_hybrid_control hybrid_control.launch.py

Optional args:
  rl_enabled:=false       → Pure model-based baseline mode
  mode:=online            → Enable online SAC training
  trajectory_type:=figure8
  world_name:=inspection_disturbance.sdf
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    pkg_andino_gz    = get_package_share_directory('andino_gz')
    pkg_hybrid_ctrl  = get_package_share_directory('andino_hybrid_control')

    params_file = os.path.join(pkg_hybrid_ctrl, 'config', 'control_params.yaml')

    # ── Launch Arguments ──────────────────────────────────────────────────────
    rl_enabled_arg = DeclareLaunchArgument(
        'rl_enabled', default_value='true',
        description='Enable RL disturbance compensation (false = baseline mode)')

    mode_arg = DeclareLaunchArgument(
        'mode', default_value='random',
        description='RL mode: random | online | pretrained')

    traj_arg = DeclareLaunchArgument(
        'trajectory_type', default_value='circle',
        description='Reference trajectory: circle | figure8 | lemniscate | square')

    radius_arg = DeclareLaunchArgument(
        'radius', default_value='2.0',
        description='Trajectory radius or scale [m]')

    speed_arg = DeclareLaunchArgument(
        'speed', default_value='0.15',
        description='Tangential speed along trajectory [m/s]')

    world_arg = DeclareLaunchArgument(
        'world_name', default_value='inspection_disturbance.sdf',
        description='Gazebo world to load')

    logging_arg = DeclareLaunchArgument(
        'enable_logging', default_value='true',
        description='Enable data logging to CSV')

    rl_enabled   = LaunchConfiguration('rl_enabled')
    mode         = LaunchConfiguration('mode')
    traj_type    = LaunchConfiguration('trajectory_type')
    radius       = LaunchConfiguration('radius')
    speed        = LaunchConfiguration('speed')
    world_name   = LaunchConfiguration('world_name')
    enable_logging = LaunchConfiguration('enable_logging')

    # ── 1. Base Gazebo simulation ─────────────────────────────────────────────
    andino_gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_andino_gz, 'launch', 'andino_gz.launch.py')
        ),
        launch_arguments={
            'world_name': world_name,
            'rviz': 'True',
            'ros_bridge': 'True',
            'nav2': 'False',
            'autostart': 'True',
        }.items(),
    )

    # ── 2. Trajectory Generator ───────────────────────────────────────────────
    # Delayed 5s to allow Gazebo + robot to initialise fully
    traj_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='andino_hybrid_control',
                executable='trajectory_generator',
                name='trajectory_generator',
                output='screen',
                parameters=[
                    params_file,
                    {'trajectory_type': traj_type},
                    {'radius': radius},
                    {'speed': speed},
                ],
            )
        ]
    )

    # ── 3. Nonlinear Controller ───────────────────────────────────────────────
    nonlinear_ctrl_node = TimerAction(
        period=5.5,
        actions=[
            Node(
                package='andino_hybrid_control',
                executable='nonlinear_controller',
                name='nonlinear_controller',
                output='screen',
                parameters=[params_file],
            )
        ]
    )

    # ── 4. RL Disturbance Compensator ─────────────────────────────────────────
    rl_node = TimerAction(
        period=5.5,
        actions=[
            Node(
                package='andino_hybrid_control',
                executable='rl_compensator',
                name='rl_disturbance_compensator',
                output='screen',
                parameters=[
                    params_file,
                    {'mode': mode},
                ],
            )
        ]
    )

    # ── 5. Hybrid Controller ──────────────────────────────────────────────────
    hybrid_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='andino_hybrid_control',
                executable='hybrid_controller',
                name='hybrid_controller',
                output='screen',
                parameters=[
                    params_file,
                    {'rl_enabled': rl_enabled},
                ],
            )
        ]
    )

    # ── 6. Performance Monitor ────────────────────────────────────────────────
    monitor_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='andino_hybrid_control',
                executable='performance_monitor',
                name='performance_monitor',
                output='screen',
                parameters=[params_file],
            )
        ]
    )
    
    # ── 7. Data Logger ────────────────────────────────────────────────────────
    logger_node = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='andino_hybrid_control',
                executable='data_logger',
                name='data_logger',
                output='screen',
                parameters=[
                    params_file,
                    {'run_label': traj_type},
                ],
                condition=IfCondition(enable_logging)
            )
        ]
    )

    return LaunchDescription([
        rl_enabled_arg,
        mode_arg,
        traj_arg,
        radius_arg,
        speed_arg,
        world_arg,
        logging_arg,
        andino_gz_launch,
        traj_node,
        nonlinear_ctrl_node,
        rl_node,
        hybrid_node,
        monitor_node,
        logger_node,
    ])
