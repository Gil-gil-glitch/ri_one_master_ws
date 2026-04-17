import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from open_manipulator_msgs.srv import SetJointPosition

class DialogueManager(Node):
    def __init__(self):
        super().__init__('dialogue_manager')
        self.sub_action = self.create_subscription(String, '/task_action', self.action_cb, 10)
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)
        self.arm_client = self.create_client(SetJointPosition, 'goal_joint_space_path')

    def action_cb(self, msg):
        data = json.loads(msg.data)
        if data.get("action") == "POINT_TO_SEAT":
            # アームで席を指差す
            self.get_logger().info("Pointing to seat with arm...")
            self.move_arm([0.6, 0.2, -0.2, 0.0]) # 席を指す角度（要調整）
            
            # 発話
            guest_name = data.get("data", {}).get("name", "guest")
            msg_tts = String()
            msg_tts.data = f"I found a seat for you, {guest_name}. Please take a seat here."
            self.pub_tts.publish(msg_tts)

    def move_arm(self, positions):
        if not self.arm_client.wait_for_service(timeout_sec=1.0):
            return
        req = SetJointPosition.Request()
        req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4']
        req.joint_position.position = positions
        req.path_time = 2.0
        self.arm_client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = DialogueManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()