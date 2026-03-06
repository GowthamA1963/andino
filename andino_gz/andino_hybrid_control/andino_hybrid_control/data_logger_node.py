#!/usr/bin/env python3
"""
Data Logger Node
================
Records all control signals and metrics to a timestamped CSV file.
Run this during a simulation trial, then use plot_results.py to generate
paper figures.

CSV columns:
  time, x, y, ref_x, ref_y, theta, ref_theta,
  v_model, w_model, v_rl, w_rl, v_total, w_total,
  e_pos, e_theta, rms_pos, rms_head, smoothness, rl_norm

Usage:
  ros2 run andino_hybrid_control data_logger
  # Saves to ~/hybrid_control_logs/run_<timestamp>.csv
"""

import os
import csv
import math
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray


class DataLoggerNode(Node):
    def __init__(self):
        super().__init__('data_logger')

        self.declare_parameter('output_dir',
                               os.path.expanduser('~/hybrid_control_logs'))
        self.declare_parameter('run_label', 'hybrid')  # 'hybrid' or 'baseline'

        out_dir   = self.get_parameter('output_dir').value
        run_label = self.get_parameter('run_label').value
        os.makedirs(out_dir, exist_ok=True)

        ts = time.strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(out_dir, f'run_{run_label}_{ts}.csv')

        self._file = open(log_path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            'time', 'x', 'y', 'ref_x', 'ref_y', 'theta', 'ref_theta',
            'v_model', 'w_model', 'v_rl', 'w_rl', 'v_total', 'w_total',
            'e_pos', 'e_theta', 'rms_pos', 'rms_head', 'smoothness', 'rl_norm'
        ])

        # ── State ─────────────────────────────────────────────────────────────
        self.x = self.y = self.theta = 0.0
        self.ref_x = self.ref_y = self.ref_theta = 0.0
        self.v_model = self.w_model = 0.0
        self.v_rl = self.w_rl = 0.0
        self.v_total = self.w_total = 0.0
        self.rms_pos = self.rms_head = self.smoothness = self.rl_norm = 0.0
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(PoseStamped, '/reference_pose', self._ref_cb, 10)
        self.create_subscription(Twist, '/cmd_vel_model', self._model_cb, 10)
        self.create_subscription(Twist, '/cmd_vel_rl_correction', self._rl_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(Float64MultiArray, '/performance_metrics',
                                 self._metrics_cb, 10)

        self.create_timer(0.1, self._log_cb)  # 10 Hz logging
        self.get_logger().info(f"DataLogger writing to: {log_path}")

    def _odom_cb(self, m):
        self.x = m.pose.pose.position.x
        self.y = m.pose.pose.position.y
        q = m.pose.pose.orientation
        self.theta = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

    def _ref_cb(self, m):
        self.ref_x = m.pose.position.x
        self.ref_y = m.pose.position.y
        q = m.pose.orientation
        self.ref_theta = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

    def _model_cb(self, m):
        self.v_model = m.linear.x; self.w_model = m.angular.z

    def _rl_cb(self, m):
        self.v_rl = m.linear.x; self.w_rl = m.angular.z

    def _cmd_cb(self, m):
        self.v_total = m.linear.x; self.w_total = m.angular.z

    def _metrics_cb(self, m):
        d = m.data
        if len(d) >= 4:
            self.rms_pos, self.rms_head, self.smoothness, self.rl_norm = \
                d[0], d[1], d[2], d[3]

    def _log_cb(self):
        t = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        e_pos   = math.hypot(self.ref_x - self.x, self.ref_y - self.y)
        e_theta = abs((self.ref_theta - self.theta + math.pi) % (2*math.pi) - math.pi)
        self._writer.writerow([
            f'{t:.3f}', f'{self.x:.4f}', f'{self.y:.4f}',
            f'{self.ref_x:.4f}', f'{self.ref_y:.4f}',
            f'{math.degrees(self.theta):.2f}', f'{math.degrees(self.ref_theta):.2f}',
            f'{self.v_model:.4f}', f'{self.w_model:.4f}',
            f'{self.v_rl:.4f}', f'{self.w_rl:.4f}',
            f'{self.v_total:.4f}', f'{self.w_total:.4f}',
            f'{e_pos:.4f}', f'{math.degrees(e_theta):.2f}',
            f'{self.rms_pos:.4f}', f'{math.degrees(self.rms_head):.2f}',
            f'{self.smoothness:.4f}', f'{self.rl_norm:.4f}'
        ])
        self._file.flush()

    def destroy_node(self):
        self._file.close()
        self.get_logger().info("DataLogger: CSV file closed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
