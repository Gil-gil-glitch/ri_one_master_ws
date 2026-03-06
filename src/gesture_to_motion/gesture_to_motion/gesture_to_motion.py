import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class GestureToMotion(Node):

    def __init__(self):
        super().__init__('gesture_to_motion')

        # Subscribe to gesture topic
        self.subscription = self.create_subscription(
            String,
            '/gesture',
            self.gesture_callback,
            10)

        # Publish to correct Kobuki velocity topic
        self.cmd_pub = self.create_publisher(
            Twist,
            '/commands/velocity',
            10)

        # Current velocity command
        self.current_twist = Twist()

        # Publish velocity continuously at 10 Hz
        self.timer = self.create_timer(
            0.1,
            self.publish_velocity
        )

        self.get_logger().info("Gesture to Motion Node Started")

    def stop_robot(self):

        self.current_twist.linear.x = 0.0
        self.current_twist.angular.z = 0.0

        self.get_logger().info("Robot stopped")



    def gesture_callback(self, msg):

        self.get_logger().info(f"Gesture received: {msg.data} ")

        if msg.data == "open_palm":
            self.get_logger().info("Open palm detected. Moving forward.")
            self.current_twist.linear.x = 0.3
            self.current_twist.angular.z = 0.0


            self.create_timer(1.0, self.stop_robot)

        elif msg.data == "fist":
            self.get_logger().info("Fist detected. Stopping.")
            self.current_twist.linear.x = 0.0
            self.current_twist.angular.z = 0.0



    def publish_velocity(self):
        self.cmd_pub.publish(self.current_twist)


def main(args=None):
    rclpy.init(args=args)
    node = GestureToMotion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()