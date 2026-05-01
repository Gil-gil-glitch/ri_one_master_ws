import json
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import String

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # Subscribers / Publishers
        self.sub_vision = self.create_subscription(String, '/receptionist/detections', self.vision_cb, 10)
        self.sub_profile = self.create_subscription(String, '/person_profile', self.profile_cb, 10)
        self.sub_action_status = self.create_subscription(String, '/action_status', self.action_status_cb, 10)
        self.pub_nlp_trigger = self.create_publisher(String, '/nlp_instruction', 10)
        self.pub_action = self.create_publisher(String, '/task_action', 10)

        # 状態管理
        self.state = "WAITING_FOR_GUEST"
        self.guest_count = 0
        self.last_vision_status = "searching"
        self.last_reception_time = 0.0
        self.cooldown_period = 5.0

        self.get_logger().info("Task Planner Node started (Field-less mode).")

    def _filter_guests(self, people: list) -> list:
        """
        領域判定を行わず、純粋に検知された人の中からゲストを抽出する
        """
        valid = []
        for p in people:
            # 1. identity で Chris (ホスト) を除外
            identity = p.get("identity") or {}
            if isinstance(identity, dict) and identity.get("name") == "Chris":
                self.get_logger().info("Filtered: Chris by identity.")
                continue
            
            # 2. 【変更点】領域座標のチェックを完全にスキップ
            # 視界に入っている人は全員候補とする
            
            valid.append(p)
        return valid

    def vision_cb(self, msg):
        if self.state != "WAITING_FOR_GUEST":
            return
        if (time.time() - self.last_reception_time) < self.cooldown_period:
            return

        data = json.loads(msg.data)
        current_status = data.get("status")

        if current_status == "guest_arrived" and self.last_vision_status == "searching":
            people = data.get("people", [])
            if not people:
                self.last_vision_status = current_status
                return

            valid_guests = self._filter_guests(people)
            if not valid_guests:
                self.last_vision_status = current_status
                return

            # フィルタを通った最初の人物をターゲットにする
            target_guest = valid_guests[0]
            bbox = target_guest.get("bbox", [])

            self.get_logger().info("Guest detected! Instructing robot to approach.")
            self.state = "APPROACHING_GUEST"

            instruction = {
                "action": "MOVE_FORWARD_TO_GUEST",
                "data": {"bbox": bbox}
            }
            self.pub_action.publish(String(data=json.dumps(instruction)))

        self.last_vision_status = current_status

    def profile_cb(self, msg):
        profile = json.loads(msg.data)
        self.guest_count += 1
        self.state = "GOING_TO_HOST"
        instruction = {
            "action": "MOVE_TO_HOST",
            "data": {
                "name": profile.get("name"),
                "drink": profile.get("drink"),
                "guest_num": self.guest_count,
            }
        }
        self.pub_action.publish(String(data=json.dumps(instruction)))

    def action_status_cb(self, msg):
        if msg.data == "ARRIVED_AT_GUEST":
            self.state = "RECEPTION"
            self.pub_nlp_trigger.publish(String(data="START_GUEST_RECEPTION"))
        elif msg.data == "COMPLETED_GUEST_MANAGEMENT":
            self.last_reception_time = time.time()
            if self.guest_count < 2:
                self.state = "RETURNING_TO_DOOR"
                self.pub_action.publish(String(data=json.dumps({"action": "MOVE_TO_DOOR"})))
            else:
                self.state = "FINISHED"
        elif msg.data == "ARRIVED_AT_DOOR":
            self.state = "WAITING_FOR_GUEST"

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TaskPlannerNode())
    rclpy.shutdown()