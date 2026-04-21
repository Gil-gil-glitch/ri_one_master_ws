import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # Subscribers
        self.sub_vision = self.create_subscription(String, '/receptionist/detections', self.vision_cb, 10)
        self.sub_profile = self.create_subscription(String, '/person_profile', self.profile_cb, 10)
        self.sub_action_status = self.create_subscription(String, '/action_status', self.action_status_cb, 10)

        # Publishers
        self.pub_nlp_trigger = self.create_publisher(String, '/nlp_instruction', 10)
        self.pub_action = self.create_publisher(String, '/task_action', 10)

        # 状態管理
        self.state = "WAITING_FOR_GUEST" # WAITING_FOR_GUEST, RECEPTION, GOING_TO_HOST, RETURNING_TO_DOOR
        self.guest_count = 0
        self.last_vision_status = "searching"

        self.get_logger().info("Advanced Task Planner Node started.")

    def vision_cb(self, msg):
        """人が来たら受付を開始する"""
        if self.state != "WAITING_FOR_GUEST":
            return

        data = json.loads(msg.data)
        if data.get("status") == "guest_arrived" and self.last_vision_status == "searching":
            self.get_logger().info("Guest detected! Starting reception...")
            self.state = "RECEPTION"
            trigger_msg = String()
            trigger_msg.data = "START_GUEST_RECEPTION"
            self.pub_nlp_trigger.publish(trigger_msg)
        
        self.last_vision_status = data.get("status")

    def profile_cb(self, msg):
        """NLPで名前が確定したら移動指示を出す"""
        profile = json.loads(msg.data)
        self.guest_count += 1
        self.state = "GOING_TO_HOST"

        # 行き先を決定 (1人目と2人目で紹介内容を変える等の拡張が可能)
        instruction = {
            "action": "MOVE_TO_HOST",
            "data": {
                "name": profile.get("name"),
                "drink": profile.get("drink"),
                "guest_num": self.guest_count
            }
        }
        self.pub_action.publish(String(data=json.dumps(instruction)))

    def action_status_cb(self, msg):
        """DialogueManagerから『到着した』『紹介が終わった』という報告を受ける"""
        if msg.data == "COMPLETED_GUEST_MANAGEMENT":
            if self.guest_count < 2:
                self.get_logger().info("Returning to door for the next guest...")
                self.state = "RETURNING_TO_DOOR"
                self.pub_action.publish(String(data=json.dumps({"action": "MOVE_TO_DOOR"})))
            else:
                self.get_logger().info("All tasks completed!")
                self.state = "FINISHED"

        elif msg.data == "ARRIVED_AT_DOOR":
            self.state = "WAITING_FOR_GUEST"
            self.get_logger().info("Ready for the next guest at the door.")

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TaskPlannerNode())
    rclpy.shutdown()