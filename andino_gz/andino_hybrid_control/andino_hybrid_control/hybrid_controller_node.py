#!/usr/bin/env python3
"""
Hybrid Controller Node
======================
Sums model-based control (u_model) and RL correction (u_RL) to produce
the final command sent to the robot:

    u_total = u_model + u_RL

This is the core of the additive hybrid architecture described in the paper
(Figure 3). The RL correction is bounded, so the nonlinear controller's
stability properties are preserved by the additive structure.

Subscribes:
    /cmd_vel_model          (geometry_msgs/Twist) — from nonlinear_controller
    /cmd_vel_rl_correction  (geometry_msgs/Twist) — from rl_compensator

Publishes:
    /cmd_vel                (geometry_msgs/Twist) — to Gazebo via ros_gz_bridge
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class HybridControllerNode(Node):
    """Additive hybrid controller: u = u_model + u_RL."""

    def __init__(self):
        super().__init__('hybrid_controller')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('v_max',      0.3)   # [m/s]   — robot hard limit
        self.declare_parameter('omega_max',  1.5)   # [rad/s] — robot hard limit
        self.declare_parameter('rl_enabled', True)  # disable RL for baseline runs

        self.v_max      = self.get_parameter('v_max').value
        self.w_max      = self.get_parameter('omega_max').value
        self.rl_enabled = self.get_parameter('rl_enabled').value

        # ── Latest commands (initialised to zero-velocity) ────────────────────
        self.u_model = Twist()
        self.u_rl    = Twist()
        self.got_model = False
        self.got_rl    = False

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_model = self.create_subscription(
            Twist, '/cmd_vel_model', self._model_cb, 10)
        self.sub_rl = self.create_subscription(
            Twist, '/cmd_vel_rl_correction', self._rl_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_cmd   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_debug = self.create_publisher(
            Float64MultiArray, '/hybrid_controller_debug', 10)

        # ── Timer: publish at 20 Hz regardless of message timing ──────────────
        self.timer = self.create_timer(0.05, self._timer_cb)

        self.get_logger().info(
            f"HybridController ready. RL={'enabled' if self.rl_enabled else 'DISABLED (baseline mode)'}, "
            f"limits: v≤{self.v_max}, ω≤{self.w_max}")

    # ──────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ──────────────────────────────────────────────────────────────────────────

    def _model_cb(self, msg: Twist):
        self.u_model   = msg
        self.got_model = True

    def _rl_cb(self, msg: Twist):
        self.u_rl    = msg
        self.got_rl  = True

    # ──────────────────────────────────────────────────────────────────────────
    # Hybrid combination
    # ──────────────────────────────────────────────────────────────────────────

    def _timer_cb(self):
        if not self.got_model:
            return  # wait until controller is live

        # u_total = u_model + u_RL  (only if RL is enabled)
        v_model = self.u_model.linear.x
        w_model = self.u_model.angular.z

        if self.rl_enabled and self.got_rl:
            v_total = v_model + self.u_rl.linear.x
            w_total = w_model + self.u_rl.angular.z
        else:
            v_total = v_model
            w_total = w_model

        # Hard saturation at robot physical limits
        v_total = max(-self.v_max, min(self.v_max, v_total))
        w_total = max(-self.w_max, min(self.w_max, w_total))

        # Publish to robot
        cmd = Twist()
        cmd.linear.x  = v_total
        cmd.angular.z = w_total
        self.pub_cmd.publish(cmd)

        # Debug info for PlotJuggler / monitoring
        debug = Float64MultiArray()
        debug.data = [
            v_model,                     # 0: model linear
            w_model,                     # 1: model angular
            self.u_rl.linear.x,          # 2: RL correction linear
            self.u_rl.angular.z,         # 3: RL correction angular
            v_total,                     # 4: final linear
            w_total,                     # 5: final angular
        ]
        self.pub_debug.publish(debug)


def main(args=None):
    rclpy.init(args=args)
    node = HybridControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
