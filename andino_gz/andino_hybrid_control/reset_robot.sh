#!/bin/bash
# Script to reset robot position in Gazebo Ignition
# Usage: ./reset_robot.sh

echo "Resetting Andino robot pose to (0,0)..."

ign service -s /world/gazebo_world/set_pose \
--reqtype ignition.msgs.Pose \
--repotype ignition.msgs.Boolean \
--timeout 2000 \
--req 'name: "andino", position: {x: 0, y: 0, z: 0.1}, orientation: {w: 1, x: 0, y: 0, z: 0}'

echo "Resetting simulation clock and all objects..."

ign service -s /world/gazebo_world/control \
--reqtype ignition.msgs.WorldControl \
--repotype ignition.msgs.Boolean \
--timeout 2000 \
--req 'reset: {all: true}'

echo "Done."
