#!/usr/bin/env python3
"""
Odom to TF Publisher Node

This node subscribes to the /odom topic and publishes the transform
from odom to base_link. This is a workaround for Gazebo's OdometryPublisher
plugin not reliably publishing TF transforms.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomToTFPublisher(Node):
    """Publishes TF transform from odometry messages."""

    def __init__(self):
        super().__init__('odom_to_tf_publisher')
        
        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribe to odometry topic
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )
        
        self.get_logger().info('Odom to TF Publisher node started')

    def odom_callback(self, msg: Odometry):
        """
        Callback for odometry messages.
        
        Publishes the transform from odom frame to base_link frame.
        """
        # Create transform message
        t = TransformStamped()
        
        # Set header
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id  # odom frame
        t.child_frame_id = msg.child_frame_id    # base_link frame
        
        # Set translation from odometry pose
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        
        # Set rotation from odometry pose
        t.transform.rotation = msg.pose.pose.orientation
        
        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToTFPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
