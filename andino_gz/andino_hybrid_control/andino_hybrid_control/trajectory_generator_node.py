#!/usr/bin/env python3
"""
Trajectory Generator Node
=========================
Publishes a smooth reference trajectory for the Andino robot to follow.
Supports: circle, figure8, lemniscate, square.
The reference state [x_d, y_d, theta_d, v_d, omega_d] is published
at a fixed rate for the nonlinear controller to track.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


class TrajectoryGeneratorNode(Node):
    """Generates smooth reference trajectories for inspection robot navigation."""

    SUPPORTED_TRAJECTORIES = ['circle', 'figure8', 'lemniscate', 'square']

    def __init__(self):
        super().__init__('trajectory_generator')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('trajectory_type', 'circle')
        self.declare_parameter('radius', 2.0)          # metres
        self.declare_parameter('speed', 0.2)           # m/s along path
        self.declare_parameter('publish_rate', 20.0)   # Hz
        self.declare_parameter('loop', True)
        self.declare_parameter('center_x', 0.0)
        self.declare_parameter('center_y', 0.0)

        self.traj_type = self.get_parameter('trajectory_type').value
        self.radius    = self.get_parameter('radius').value
        self.speed     = self.get_parameter('speed').value
        self.loop      = self.get_parameter('loop').value
        self.cx        = self.get_parameter('center_x').value
        self.cy        = self.get_parameter('center_y').value
        rate           = self.get_parameter('publish_rate').value

        if self.traj_type not in self.SUPPORTED_TRAJECTORIES:
            self.get_logger().warn(
                f"Unknown trajectory '{self.traj_type}'. Defaulting to 'circle'.")
            self.traj_type = 'circle'

        # ── Publishers ────────────────────────────────────────────────────────
        # Desired pose for the controller
        self.pub_pose = self.create_publisher(PoseStamped, '/reference_pose', 10)
        # Desired velocity feedforward terms
        self.pub_vel  = self.create_publisher(TwistStamped, '/reference_velocity', 10)
        # RViz marker for visualising the full path
        self.pub_marker = self.create_publisher(Marker, '/trajectory_marker', 10)

        # ── Internal state ────────────────────────────────────────────────────
        self.t = 0.0          # parametric time along trajectory
        self.dt = 1.0 / rate  # timer period

        self.timer = self.create_timer(self.dt, self._timer_cb)
        self.get_logger().info(
            f"TrajectoryGenerator started: type={self.traj_type}, "
            f"radius={self.radius}m, speed={self.speed}m/s")

        # Publish the full path marker once at start
        self._publish_path_marker()

    # ──────────────────────────────────────────────────────────────────────────
    # Trajectory maths
    # ──────────────────────────────────────────────────────────────────────────

    def _sample(self, t):
        """
        Return (x, y, theta, v_d, omega_d) at parametric time t.
        All trajectories are arc-length parameterised by self.speed.
        """
        R = self.radius
        s = self.speed

        if self.traj_type == 'circle':
            # Angular velocity to maintain speed on circle: omega = v / R
            omega = s / R
            x     = self.cx + R * math.cos(omega * t)
            y     = self.cy + R * math.sin(omega * t)
            theta = math.atan2(-math.sin(omega * t), math.cos(omega * t)) + math.pi / 2
            v_d   = s
            w_d   = omega

        elif self.traj_type == 'figure8':
            # Lemniscate of Bernoulli variant — parametric at speed s
            # Period: 2π / ω,  ω chosen so tangential speed ≈ s
            omega = s / (R * math.sqrt(2))
            x  = self.cx + R * math.sin(omega * t)
            y  = self.cy + R * math.sin(omega * t) * math.cos(omega * t)
            # Derivatives for heading
            dx = R * omega * math.cos(omega * t)
            dy = R * omega * (math.cos(omega * t)**2 - math.sin(omega * t)**2)
            theta = math.atan2(dy, dx)
            spd   = math.hypot(dx, dy)
            # Curvature-based omega (finite diff)
            dt_  = 0.01
            x2   = self.cx + R * math.sin(omega * (t + dt_))
            y2   = self.cy + R * math.sin(omega * (t + dt_)) * math.cos(omega * (t + dt_))
            dx2  = R * omega * math.cos(omega * (t + dt_))
            dy2  = R * omega * (math.cos(omega * (t + dt_))**2 - math.sin(omega * (t + dt_))**2)
            dtheta = math.atan2(dy2, dx2) - theta
            # Wrap to [-pi, pi]
            dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
            v_d = spd
            w_d = dtheta / dt_

        elif self.traj_type == 'lemniscate':
            # Standard lemniscate: x=a·cos(t)/(1+sin²t), y=a·sin(t)cos(t)/(1+sin²t)
            omega = s / R
            denom = 1.0 + math.sin(omega * t) ** 2
            x     = self.cx + R * math.cos(omega * t) / denom
            y     = self.cy + R * math.sin(omega * t) * math.cos(omega * t) / denom
            dt_   = 0.01
            denom2 = 1.0 + math.sin(omega * (t + dt_)) ** 2
            x2  = self.cx + R * math.cos(omega * (t + dt_)) / denom2
            y2  = self.cy + R * math.sin(omega * (t + dt_)) * math.cos(omega * (t + dt_)) / denom2
            dx  = (x2 - x) / dt_
            dy  = (y2 - y) / dt_
            theta = math.atan2(dy, dx)
            v_d   = math.hypot(dx, dy)
            # Second derivative for omega
            denom3 = 1.0 + math.sin(omega * (t + 2 * dt_)) ** 2
            x3   = self.cx + R * math.cos(omega * (t + 2 * dt_)) / denom3
            y3   = self.cy + R * math.sin(omega * (t + 2 * dt_)) * math.cos(omega * (t + 2 * dt_)) / denom3
            dx2  = (x3 - x2) / dt_
            dy2  = (y3 - y2) / dt_
            theta2 = math.atan2(dy2, dx2)
            dtheta = (theta2 - theta + math.pi) % (2 * math.pi) - math.pi
            w_d = dtheta / dt_

        elif self.traj_type == 'square':
            # Square centered at (cx, cy) with side length 2*R
            # Passing through (R, 0), (0, R), (-R, 0), (0, -R) midpoints
            side    = R * 2.0
            period  = 4.0 * side / s
            phase   = (t % period) / period
            
            # Corners centered around center
            corners = [
                (self.cx + R, self.cy - R), # Bottom-Right
                (self.cx + R, self.cy + R), # Top-Right
                (self.cx - R, self.cy + R), # Top-Left
                (self.cx - R, self.cy - R), # Bottom-Left
            ]
            
            seg = int(phase * 4)
            seg = min(seg, 3)
            seg_phase = (phase * 4) - seg
            
            p0 = corners[seg]
            p1 = corners[(seg + 1) % 4]
            
            x = p0[0] + seg_phase * (p1[0] - p0[0])
            y = p0[1] + seg_phase * (p1[1] - p0[1])
            
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            theta = math.atan2(dy, dx)
            v_d = s
            w_d = 0.0
        else:
            x, y, theta, v_d, w_d = 0.0, 0.0, 0.0, 0.0, 0.0

        return x, y, theta, v_d, w_d

    # ──────────────────────────────────────────────────────────────────────────
    # Timer callback
    # ──────────────────────────────────────────────────────────────────────────

    def _timer_cb(self):
        self.t += self.dt
        x, y, theta, v_d, w_d = self._sample(self.t)

        now = self.get_clock().now().to_msg()

        # ── PoseStamped ───────────────────────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header = Header(frame_id='odom', stamp=now)
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = 0.0
        # Quaternion from yaw
        pose_msg.pose.orientation.z = math.sin(theta / 2.0)
        pose_msg.pose.orientation.w = math.cos(theta / 2.0)
        self.pub_pose.publish(pose_msg)

        # ── TwistStamped (feedforward velocities) ─────────────────────────────
        vel_msg = TwistStamped()
        vel_msg.header = Header(frame_id='odom', stamp=now)
        vel_msg.twist.linear.x  = v_d
        vel_msg.twist.angular.z = w_d
        self.pub_vel.publish(vel_msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Path visualisation marker (published once)
    # ──────────────────────────────────────────────────────────────────────────

    def _publish_path_marker(self):
        from geometry_msgs.msg import Point
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'reference_trajectory'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # line width
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 0.9
        marker.pose.orientation.w = 1.0

        # Sample 300 points around the trajectory
        steps = 300
        period = self._estimate_period()
        for i in range(steps + 1):
            t_sample = period * i / steps
            x, y, _, _, _ = self._sample(t_sample)
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.01
            marker.points.append(p)

        self.pub_marker.publish(marker)

    def _estimate_period(self):
        """Estimate period for one full loop of the trajectory."""
        R, s = self.radius, self.speed
        if self.traj_type == 'circle':
            return 2 * math.pi * R / s
        elif self.traj_type in ('figure8', 'lemniscate'):
            return 2 * math.pi * R / s * 1.5
        elif self.traj_type == 'square':
            return 4 * R * 2.0 / s
        return 60.0


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
