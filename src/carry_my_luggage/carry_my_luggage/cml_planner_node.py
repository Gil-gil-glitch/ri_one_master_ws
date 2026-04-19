import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

class CMLPlannerNode(Node):
    def __init__(self):
        super().__init__('cml_planner_node')

        # === State ===
        # START -> POINTING -> GRASPING -> WAIT_FOLLOW -> FOLLOWING -> HANDOVER -> NAVIGATION_BACK -> DONE
        self.state = "START"
        self.get_logger().info("CML Planner Node started in state: START")

        # === Subscribers ===
        # 1. Voice Commands (Start signal / Handover / Reached Car)
        self.sub_voice = self.create_subscription(
            String, '/nlp/voice_command', self.voice_cb, 10)
        
        # 2. Gesture Detection (Pointing to the bag)
        self.sub_gesture = self.create_subscription(
            String, '/gesture/pointing_target', self.gesture_cb, 10)
        
        # 3. Grasping Status
        self.sub_grasp = self.create_subscription(
            String, '/manipulator/grasp_status', self.grasp_cb, 10)

        # 4. Follow-Me Status
        self.sub_follow = self.create_subscription(
            String, '/follow_me/status', self.follow_cb, 10)

        # 5. Navigation Status
        self.sub_nav = self.create_subscription(
            String, '/navigation/status', self.nav_cb, 10)

        # === Publishers ===
        # 1. TTS / Voice Feedback
        self.pub_tts = self.create_publisher(String, '/nlp/tts_command', 10)

        # 2. Manipulator Action (Grasp, Drop)
        self.pub_manipulator = self.create_publisher(String, '/manipulator/action', 10)

        # 3. Follow-Me Control (Start, Stop)
        self.pub_follow = self.create_publisher(String, '/follow_me/control', 10)

        # 4. Navigation Control (Go to Start)
        self.pub_nav = self.create_publisher(String, '/navigation/goal', 10)

    def speak(self, text):
        msg = String()
        msg.data = text
        self.pub_tts.publish(msg)
        self.get_logger().info(f"Robot says: {text}")

    def voice_cb(self, msg):
        command = msg.data.lower()
        self.get_logger().info(f"Received voice command: {command}")

        if self.state == "START" and "start" in command:
            self.state = "POINTING"
            self.speak("Ready. Please point to the bag you want me to carry.")
        
        elif self.state == "FOLLOWING" and "reached" in command:
            # We reached the goal / car
            self.state = "HANDOVER"
            self.speak("We have reached the car. Please take the bag.")
            
            # Request manipulator to release the bag after a small delay or instantly
            self.pub_follow.publish(String(data="STOP"))
            self.pub_manipulator.publish(String(data="DROP_BAG"))

    def gesture_cb(self, msg):
        if self.state != "POINTING":
            return
        
        target_info = msg.data
        self.get_logger().info(f"Detected pointing at: {target_info}")
        self.state = "GRASPING"
        self.speak(f"I will grasp the {target_info} bag.")
        
        # Trigger manipulator to grasp the target bag
        action_msg = String()
        action_msg.data = json.dumps({"action": "GRASP", "target": target_info})
        self.pub_manipulator.publish(action_msg)

    def grasp_cb(self, msg):
        if self.state != "GRASPING" and self.state != "HANDOVER":
            return
        
        status = msg.data
        if self.state == "GRASPING":
            if status == "SUCCESS":
                self.state = "WAIT_FOLLOW"
                self.speak("I have grasped the bag. I am ready to follow you. Please start walking.")
                self.state = "FOLLOWING"
                self.pub_follow.publish(String(data="START"))
            elif status == "FAILED":
                self.speak("I failed to grasp the bag. Please point to it again.")
                self.state = "POINTING"
                
        elif self.state == "HANDOVER":
            if status == "DROPPED":
                self.speak("Going back to the starting position.")
                self.state = "NAVIGATION_BACK"
                nav_msg = String(data=json.dumps({"goal": "START_POINT"}))
                self.pub_nav.publish(nav_msg)

    def follow_cb(self, msg):
        # We might receive LOST tracking here, handling re-finding logic
        if self.state != "FOLLOWING":
            return
        
        status = msg.data
        if status == "LOST_OPERATOR":
            self.speak("I lost you. Please come closer so I can find you.")
            # Depending on how advanced the rediscovery, state loop could handle "FINDING_OPERATOR"

    def nav_cb(self, msg):
        if self.state != "NAVIGATION_BACK":
            return
        
        status = msg.data
        if status == "REACHED_GOAL":
            self.state = "DONE"
            self.speak("I have successfully returned to the starting point. Task completed.")
            self.get_logger().info("CML Task Completed successfully!")

def main(args=None):
    rclpy.init(args=args)
    node = CMLPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt. Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
