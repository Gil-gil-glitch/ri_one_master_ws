#
##  Home Task Approaching Person Node
#
#  Listens on /open_challenge_state for "approach_person".
#  Drives forward at 0.5 m/s for DRIVE_DURATION seconds then stops.
#
#  Bug fix vs original: rclpy.sleep() does not exist in ROS2.
#  A one-shot timer is used instead so the executor is never blocked.
#

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

DRIVE_DURATION = 10.0   # seconds to move forward


class HomeTaskApproachingPersonNode(Node):

    def __init__(self):
        super().__init__('home_task_approaching_person_node')

        self.subscription = self.create_subscription(
            String, '/open_challenge_state', self.state_callback, 10)

        self.publisher_ = self.create_publisher(Twist, '/commands/velocity', 10)

        self._stop_timer = None
        self.get_logger().info("Home Task Approaching Person Node started.")

    def state_callback(self, msg: String):
        if msg.data != "approach_person":
            return

        # Cancel any in-progress approach (safety).
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None

        self.get_logger().info(
            f"'approach_person' received — driving forward for {DRIVE_DURATION}s.")
        self._publish_velocity(0.5)

        # Schedule stop without blocking the executor.
        self._stop_timer = self.create_timer(DRIVE_DURATION, self._stop)

    def _stop(self):
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None

        self.get_logger().info("Approach complete — stopping.")
        self._publish_velocity(0.0)

    def _publish_velocity(self, x: float):
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HomeTaskApproachingPersonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()