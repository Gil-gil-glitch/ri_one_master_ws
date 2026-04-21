import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CMLCoordinator(Node):
    def __init__(self):
        super().__init__('cml_coordinator')

        self.state = "IDLE"

        #Publishers
        self.state_pub = self.create_publisher(String, '/cml_state', 10)

        # TTS(Publish messages robot needs to say)  
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)

        #Subscribers
        self.command_sub = self.create_subscription(String, '/voice_imperatives', self.command_callback, 10)

        self.timer = self.create_timer(1.0, self.publish_state) # Publish state at 1 Hz to keep nodes in sync
        self.get_logger().info("CML Coordinator started. State: IDLE")

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
