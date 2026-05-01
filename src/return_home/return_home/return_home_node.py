import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class ReturnHomeNode(Node):
    def __init__(self):
        super().__init__('return_home_node')

        # Publisher to tell the motors to stop
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber to watch the coordinator state
        self.state_sub = self.create_subscription(String, '/cml_state', self.state_callback, 10)

        self.get_logger().info("Return Home Node ready (Waiting for 'RETURN' or 'IDLE' state)")

    def state_callback(self, msg):
        # If the state becomes RETURN or IDLE, we force a full stop
        if msg.data in ["RETURN", "IDLE"]:
            self.get_logger().info(f"State '{msg.data}' received. Halting robot...")
            self.halt_and_shutdown()

    def halt_and_shutdown(self):
        # 1. Create a zero-velocity message
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 0.0
        stop_msg.linear.z = 0.0
        stop_msg.angular.x = 0.0
        stop_msg.angular.y = 0.0
        stop_msg.angular.z = 0.0

        # 2. Publish it multiple times to ensure the robot gets it
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_msg)
        
        self.get_logger().info("Robot halted. Shutting down node.")
        
        # 3. Shutdown rclpy
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ReturnHomeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()