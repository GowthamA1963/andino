#!/usr/bin/env python3
"""
Nonlinear Controller Node (Feedback Linearization)
===================================================
Implements model-based feedback linearization for a differential-drive robot.

The standard unicycle model is:
    ẋ = v·cos(θ)
    ẏ = v·sin(θ)
    θ̇ = ω

Feedback linearization introduces an auxiliary point at distance d ahead:
    p_x = x + d·cos(θ),   p_y = y + d·sin(θ)

Then:
    [ṗ_x]   [cos(θ)  -d·sin(θ)] [v]
    [ṗ_y] = [sin(θ)   d·cos(θ)] [ω]

Which can be inverted to get (v, ω) from desired (u1, u2):
    u1 = k1·(x_d - p_x) + ẋ_d
    u2 = k2·(y_d - p_y) + ẏ_d

This gives linear error dynamics with guaranteed exponential convergence
under the nominal (disturbance-free) model.

Publishes: /cmd_vel_model  (to be summed with RL correction in hybrid_controller)
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class NonlinearControllerNode(Node):
    """Feedback linearization trajectory tracking controller."""

    def __init__(self):
        super().__init__('nonlinear_controller')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('k1',        3.0)   # gain on x-error
        self.declare_parameter('k2',        3.0)   # gain on y-error
        self.declare_parameter('k3',        3.0)   # gain on heading error
        self.declare_parameter('ki',        0.1)   # integral gain for aux point
        self.declare_parameter('d',         0.1)   # look-ahead distance [m]
        self.declare_parameter('v_max',     0.5)   # increased for faster recovery
        self.declare_parameter('omega_max', 2.0)   # increased for faster recovery
        self.declare_parameter('publish_rate', 20.0)

        self.k1    = self.get_parameter('k1').value
        self.k2    = self.get_parameter('k2').value
        self.k3    = self.get_parameter('k3').value
        self.ki    = self.get_parameter('ki').value
        self.d     = self.get_parameter('d').value
        self.v_max = self.get_parameter('v_max').value
        self.w_max = self.get_parameter('omega_max').value
        rate       = self.get_parameter('publish_rate').value

        # ── State ─────────────────────────────────────────────────────────────
        self.x = 0.0; self.y = 0.0; self.theta = 0.0
        self.v_curr = 0.0; self.w_curr = 0.0

        self.ref_x = 0.0; self.ref_y = 0.0; self.ref_theta = 0.0
        self.ref_vd = 0.0; self.ref_wd = 0.0
        self.got_odom = False
        self.got_ref  = False
        
        # ── Error Integrators ─────────────────────────────────────────────────
        self.e_px_int = 0.0
        self.e_py_int = 0.0
        self.dt = 1.0 / rate

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.sub_ref_pose = self.create_subscription(
            PoseStamped, '/reference_pose', self._ref_pose_cb, 10)
        self.sub_ref_vel = self.create_subscription(
            TwistStamped, '/reference_velocity', self._ref_vel_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_cmd   = self.create_publisher(Twist, '/cmd_vel_model', 10)
        self.pub_error = self.create_publisher(Float64MultiArray, '/tracking_error', 10)
        self.pub_error_marker = self.create_publisher(Marker, '/error_marker', 10)

        # ── Control timer ─────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / rate, self._control_cb)
        self.get_logger().info(
            f"NonlinearController started: k1={self.k1}, k2={self.k2}, "
            f"k3={self.k3}, d={self.d}m")

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # Yaw from quaternion (z, w components sufficient for 2D)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)
        self.v_curr = msg.twist.twist.linear.x
        self.w_curr = msg.twist.twist.angular.z
        self.got_odom = True

    def _ref_pose_cb(self, msg: PoseStamped):
        self.ref_x = msg.pose.position.x
        self.ref_y = msg.pose.position.y
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.ref_theta = math.atan2(siny_cosp, cosy_cosp)
        self.got_ref = True

    def _ref_vel_cb(self, msg: TwistStamped):
        self.ref_vd = msg.twist.linear.x
        self.ref_wd = msg.twist.angular.z

    # ──────────────────────────────────────────────────────────────────────────
    # Core control law — Feedback Linearization
    # ──────────────────────────────────────────────────────────────────────────

    def _control_cb(self):
        if not (self.got_odom and self.got_ref):
            return

        # ── Auxiliary look-ahead point ────────────────────────────────────────
        # p = (x + d·cos(θ),  y + d·sin(θ))
        px = self.x + self.d * math.cos(self.theta)
        py = self.y + self.d * math.sin(self.theta)

        # ── Desired auxiliary point (Ref point shifted by d) ──────────────────
        # To make the CENTRE follow (ref_x, ref_y), the LOOK-AHEAD point p
        # must follow (ref_x + d·cos(ref_theta), ref_y + d·sin(ref_theta))
        ref_px = self.ref_x + self.d * math.cos(self.ref_theta)
        ref_py = self.ref_y + self.d * math.sin(self.ref_theta)

        e_px = ref_px - px
        e_py = ref_py - py

        # ── Feed-forward auxiliary velocities ─────────────────────────────────
        # ṗ_ref = [v_ref·cos(θ_ref) - d·ω_ref·sin(θ_ref)]
        #       [v_ref·sin(θ_ref) + d·ω_ref·cos(θ_ref)]
        vd_px = self.ref_vd * math.cos(self.ref_theta) - self.d * self.ref_wd * math.sin(self.ref_theta)
        vd_py = self.ref_vd * math.sin(self.ref_theta) + self.d * self.ref_wd * math.cos(self.ref_theta)

        # ── Desired auxiliary velocities (PI Control) ─────────────────────────
        # Feed-forward + Proportional + Integral
        self.e_px_int += e_px * self.dt
        self.e_py_int += e_py * self.dt
        
        # Anti-windup (Clamped to 1.0 m/s contribution)
        self.e_px_int = max(-1.0, min(1.0, self.e_px_int))
        self.e_py_int = max(-1.0, min(1.0, self.e_py_int))

        u1 = vd_px + self.k1 * e_px + self.ki * self.e_px_int
        u2 = vd_py + self.k2 * e_py + self.ki * self.e_py_int

        # ── Invert feedback-linearised model ──────────────────────────────────
        # [v]   [ cos(θ)   sin(θ)] [u1]
        # [ω] = [-sin(θ)/d cos(θ)/d] [u2]
        c, s = math.cos(self.theta), math.sin(self.theta)
        v =  c * u1 + s * u2
        w = (-s * u1 + c * u2) / self.d

        # ── Optional additional Heading correction ───────────────────────────
        e_theta = wrap_angle(self.ref_theta - self.theta)
        w += self.k3 * e_theta

        # ── Saturate ──────────────────────────────────────────────────────────
        v = max(-self.v_max, min(self.v_max, v))
        w = max(-self.w_max, min(self.w_max, w))

        # ── Publish model control command ─────────────────────────────────────
        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)

        # ── Publish tracking errors ───────────────────────────────────────────
        e_pos = math.hypot(self.ref_x - self.x, self.ref_y - self.y)
        err_msg = Float64MultiArray()
        err_msg.data = [e_pos, abs(e_theta), abs(e_px), abs(e_py)]
        self.pub_error.publish(err_msg)

        # ── RViz arrow marker: robot → reference ─────────────────────────────
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'tracking_error'
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.02
        m.scale.y = 0.05
        m.color.r = 1.0
        m.color.g = 0.3
        m.color.b = 0.0
        m.color.a = 0.9
        start = Point(x=self.x, y=self.y, z=0.1)
        end   = Point(x=self.ref_x, y=self.ref_y, z=0.1)
        m.points = [start, end]
        self.pub_error_marker.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = NonlinearControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
