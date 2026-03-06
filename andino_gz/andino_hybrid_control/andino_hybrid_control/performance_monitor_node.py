#!/usr/bin/env python3
"""
Performance Monitor Node
========================
Computes and logs evaluation metrics for the hybrid control framework.
Publishes metrics as Float64MultiArray for recording/plotting.

Metrics:
  0: RMS position error    [m]       — lower is better
  1: RMS heading error     [rad]     — lower is better
  2: Control smoothness    [m/s²]    — rate of change of cmd_vel (lower = smoother)
  3: RL correction norm    [m/s]     — magnitude of RL contribution
  4: Mean position error   [m]       — running mean
  5: Total steps                     — counter
"""

import math
import collections
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


class PerformanceMonitorNode(Node):
    """Computes inspection-specific performance metrics."""

    def __init__(self):
        super().__init__('performance_monitor')

        self.declare_parameter('window_size', 200)   # steps for rolling RMS
        self.declare_parameter('log_interval', 10.0) # seconds between terminal logs

        win  = self.get_parameter('window_size').value
        log_int = self.get_parameter('log_interval').value

        # ── Ring buffers ──────────────────────────────────────────────────────
        self.pos_errors    = collections.deque(maxlen=win)
        self.head_errors   = collections.deque(maxlen=win)
        self.cmd_vel_hist  = collections.deque(maxlen=5)   # for smoothness

        # ── State ─────────────────────────────────────────────────────────────
        self.x = 0.0; self.y = 0.0; self.theta = 0.0
        self.ref_x = 0.0; self.ref_y = 0.0; self.ref_theta = 0.0
        self.last_cmd_v = 0.0; self.last_cmd_w = 0.0
        self.got_odom = False; self.got_ref = False

        self.total_steps = 0
        self.rl_correction_v = 0.0
        self.rl_correction_w = 0.0

        # disturbance recovery tracking
        self.in_disturbance = False
        self.recovery_start = None
        self.recovery_times = []

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(PoseStamped, '/reference_pose', self._ref_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(
            Twist, '/cmd_vel_rl_correction', self._rl_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_metrics = self.create_publisher(
            Float64MultiArray, '/performance_metrics', 10)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(0.05,   self._update_cb)   # 20 Hz metric update
        self.create_timer(log_int, self._log_cb)     # terminal summary

        self.get_logger().info("PerformanceMonitor started.")

    # ──────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ──────────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny, cosy)
        self.got_odom = True

    def _ref_cb(self, msg: PoseStamped):
        self.ref_x = msg.pose.position.x
        self.ref_y = msg.pose.position.y
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.ref_theta = math.atan2(siny, cosy)
        self.got_ref = True

    def _cmd_cb(self, msg: Twist):
        self.cmd_vel_hist.append((msg.linear.x, msg.angular.z))
        self.last_cmd_v = msg.linear.x
        self.last_cmd_w = msg.angular.z

    def _rl_cb(self, msg: Twist):
        self.rl_correction_v = msg.linear.x
        self.rl_correction_w = msg.angular.z

    # ──────────────────────────────────────────────────────────────────────────
    # Metric update
    # ──────────────────────────────────────────────────────────────────────────

    def _update_cb(self):
        if not (self.got_odom and self.got_ref):
            return

        self.total_steps += 1

        # ── Position and heading errors ───────────────────────────────────────
        e_pos = math.hypot(self.ref_x - self.x, self.ref_y - self.y)
        e_head = abs((self.ref_theta - self.theta + math.pi) % (2 * math.pi) - math.pi)

        self.pos_errors.append(e_pos ** 2)
        self.head_errors.append(e_head ** 2)

        # ── Control smoothness: mean |Δv| between consecutive steps ──────────
        smoothness = 0.0
        if len(self.cmd_vel_hist) >= 2:
            dvs = [abs(self.cmd_vel_hist[i][0] - self.cmd_vel_hist[i-1][0])
                   + abs(self.cmd_vel_hist[i][1] - self.cmd_vel_hist[i-1][1])
                   for i in range(1, len(self.cmd_vel_hist))]
            smoothness = sum(dvs) / len(dvs)

        # ── RMS metrics ───────────────────────────────────────────────────────
        rms_pos  = math.sqrt(sum(self.pos_errors)  / len(self.pos_errors))
        rms_head = math.sqrt(sum(self.head_errors) / len(self.head_errors))
        mean_pos = sum(math.sqrt(e) for e in self.pos_errors) / len(self.pos_errors)

        rl_norm = math.hypot(self.rl_correction_v, self.rl_correction_w)

        # ── Publish ───────────────────────────────────────────────────────────
        msg = Float64MultiArray()
        msg.data = [
            rms_pos,
            rms_head,
            smoothness,
            rl_norm,
            mean_pos,
            float(self.total_steps),
        ]
        self.pub_metrics.publish(msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Terminal summary
    # ──────────────────────────────────────────────────────────────────────────

    def _log_cb(self):
        if not self.pos_errors:
            return

        rms_pos  = math.sqrt(sum(self.pos_errors)  / len(self.pos_errors))
        rms_head = math.sqrt(sum(self.head_errors) / len(self.head_errors))
        rl_norm  = math.hypot(self.rl_correction_v, self.rl_correction_w)

        self.get_logger().info(
            f"\n{'='*56}\n"
            f"  PERFORMANCE METRICS  (steps={self.total_steps})\n"
            f"{'='*56}\n"
            f"  RMS Position Error  : {rms_pos:.4f} m\n"
            f"  RMS Heading Error   : {math.degrees(rms_head):.2f} deg\n"
            f"  RL Correction Norm  : {rl_norm:.4f} m/s\n"
            f"  Current Pos Error   : {math.hypot(self.ref_x-self.x, self.ref_y-self.y):.4f} m\n"
            f"{'='*56}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PerformanceMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
