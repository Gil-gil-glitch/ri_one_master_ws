import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CMLCoordinator(Node):
    def __init__(self):
        super().__init__('cml_coordinator')
        self.state = "IDLE"
        self.target_bag = "NONE" # "LEFT" or "RIGHT"

        self.state_pub = self.create_publisher(String, '/cml_state', 10)
        self.target_bag_pub = self.create_publisher(String, '/target_bag', 10)
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)

        self.command_sub = self.create_subscription(String, '/voice_imperatives', self.command_callback, 10)
        self.gesture_sub = self.create_subscription(String, '/gesture', self.gesture_callback, 10)
        
        # Listen for pickup completion to transition to follow
        self.pickup_status_sub = self.create_subscription(String, '/pickup_status', self.pickup_status_callback, 10)

        self.timer = self.create_timer(1.0, self.publish_state)
        self.get_logger().info("CML Coordinator started. State: IDLE")

    def gesture_callback(self, msg):
        if self.state == "IDLE":
            if msg.data == "pointing_left":
                self.start_pickup("LEFT")
            elif msg.data == "pointing_right":
                self.start_pickup("RIGHT")

    def start_pickup(self, direction):
        self.state = "PICKUP"
        self.target_bag = direction
        self.get_logger().info(f"Switching to PICKUP. Target: {direction}")
        self.say(f"I will pick up the bag on the {direction.lower()}.")
        
        bag_msg = String()
        bag_msg.data = self.target_bag
        self.target_bag_pub.publish(bag_msg)
        self.publish_state()

    def pickup_status_callback(self, msg):
        if self.state == "PICKUP" and msg.data == "DONE":
            self.state = "FOLLOW"
            self.get_logger().info("Pickup complete. Switching to FOLLOW")
            self.say("I have the bag. Please start walking, I will follow you.")
            self.publish_state()

    def command_callback(self, msg):
        if msg.data == "following":
            self.state = "FOLLOW"
            self.get_logger().info("Switching to FOLLOW")
            self.say("I'm ready. Please start walking, I will follow you.")

        elif msg.data == "returning":
            self.state = "RETURN"
            self.get_logger().info("Switching to RETURN")
            self.say("I will return to the start point now.")

        self.publish_state()

    def publish_state(self):
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

    def say(self, text):
        """Make Robot Speak"""
        self.get_logger().info(f"ROBOT SAYS: {text}")
        msg = String()
        msg.data = text
        self.pub_tts.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CMLCoordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()