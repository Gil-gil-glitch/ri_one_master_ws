import json
import math
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import String


class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # Subscribers
        self.sub_vision = self.create_subscription(
            String, '/receptionist/detections', self.vision_cb, 10)
        self.sub_profile = self.create_subscription(
            String, '/person_profile', self.profile_cb, 10)
        self.sub_action_status = self.create_subscription(
            String, '/action_status', self.action_status_cb, 10)

        # Publishers
        self.pub_nlp_trigger = self.create_publisher(String, '/nlp_instruction', 10)
        self.pub_action = self.create_publisher(String, '/task_action', 10)

        self.FIELD_X_MIN = -100
        self.FIELD_X_MAX =  100.69
        self.FIELD_Y_MIN = -100.51
        self.FIELD_Y_MAX =  100.15
        self.FIELD_MARGIN = 0.3

        # 状態管理
        self.state = "WAITING_FOR_GUEST"
        self.guest_count = 0
        self.last_vision_status = "searching"
        self.current_bonus_data = {}

        # クールダウン
        self.last_reception_time = 0.0
        self.cooldown_period = 5.0

        self.get_logger().info("Task Planner Node started.")

    def _filter_guests(self, people: list) -> list:
        # Simple filter for the first few guests
        valid = []
        for p in people:
            identity = p.get("identity") or {}
            if isinstance(identity, dict) and identity.get("name") == "Chris":
                continue
            valid.append(p)
        return valid

    def vision_cb(self, msg):
        data = json.loads(msg.data)
        
        if self.state == "BONUS_VISION_DETECTING":
            # Just capture features of the first person we see (assuming we are facing Judge 1)
            people = data.get("people", [])
            if people:
                attrs = people[0].get("attributes", {})
                self.current_bonus_data["vision"] = attrs
                self.get_logger().info(f"Bonus Vision Captured: {attrs}")
                self.state = "BONUS_VISION_MOVE_TO_JUDGE2"
                self.pub_action.publish(String(data=json.dumps({"action": "MOVE_TO_FACE_JUDGE_2"})))
            return

        if self.state != "WAITING_FOR_GUEST":
            return
        if (time.time() - self.last_reception_time) < self.cooldown_period:
            return

        current_status = data.get("status")
        if current_status == "guest_arrived" and self.last_vision_status == "searching":
            people = data.get("people", [])
            valid_guests = self._filter_guests(people)
            if valid_guests:
                self.get_logger().info("Guest detected! Sending forward movement...")
                self.state = "APPROACHING_GUEST"
                self.pub_action.publish(String(data=json.dumps({
                    "action": "MOVE_FORWARD_TO_GUEST",
                    "data": {"bbox": valid_guests[0].get("bbox")}
                })))

        self.last_vision_status = current_status

    def profile_cb(self, msg):
        profile = json.loads(msg.data)
        
        # Check if it's the end of bonus voice
        if profile.get("type") == "BONUS_VOICE_DONE":
            self.current_bonus_data["voice"] = profile
            self.state = "BONUS_VOICE_MOVE_TO_JUDGE2"
            self.pub_action.publish(String(data=json.dumps({"action": "MOVE_TO_FACE_JUDGE_2"})))
            return

        # Normal reception profile
        self.guest_count += 1
        self.state = "GOING_TO_HOST"
        self.pub_action.publish(String(data=json.dumps({
            "action": "MOVE_TO_HOST",
            "data": {
                "name": profile.get("name"),
                "drink": profile.get("drink"),
                "guest_num": self.guest_count,
            }
        })))

    def action_status_cb(self, msg):
        if msg.data == "ARRIVED_AT_GUEST":
            if self.state == "APPROACHING_GUEST":
                self.state = "RECEPTION"
                self.pub_nlp_trigger.publish(String(data="START_GUEST_RECEPTION"))

        elif msg.data == "COMPLETED_GUEST_MANAGEMENT":
            self.last_reception_time = time.time()
            if self.guest_count == 1:
                self.state = "RETURNING_TO_DOOR"
                self.pub_action.publish(String(data=json.dumps({"action": "MOVE_TO_DOOR"})))
            elif self.guest_count == 2:
                self.get_logger().info("Second guest seated. Starting BONUS phase.")
                self.state = "BONUS_START"
                # Phase 1: Face Judge 1 for Vision
                self.pub_action.publish(String(data=json.dumps({"action": "MOVE_TO_FACE_JUDGE_1"})))

        elif msg.data == "ARRIVED_AT_DOOR":
            self.state = "WAITING_FOR_GUEST"

        # --- Bonus Phase Transitions ---
        elif msg.data == "FACING_JUDGE_1":
            if self.state == "BONUS_START":
                self.state = "BONUS_VISION_DETECTING"
                # Wait for vision_cb to pick up features
            elif self.state == "BONUS_VOICE_MOVE_TO_JUDGE1":
                self.state = "BONUS_VOICE_ASKING"
                self.pub_nlp_trigger.publish(String(data="START_BONUS_VOICE"))

        elif msg.data == "FACING_JUDGE_2":
            if self.state == "BONUS_VISION_MOVE_TO_JUDGE2":
                self.state = "BONUS_SAYING_VISION"
                attrs = self.current_bonus_data.get("vision", {})
                shirt = attrs.get("Clothing color (Tops) >>", "unknown")
                glasses = attrs.get("Wears Glasses >>", "no")
                hair = attrs.get("Hair Length >>", "short")
                carrying = attrs.get("Carrying Item >>", "no")
                
                text = f"The first judge is wearing a {shirt} shirt. They "
                text += "are wearing glasses" if glasses == "Yes" else "are not wearing glasses"
                text += f". They have {hair} hair. "
                text += f"They are carrying {carrying}." if "Yes" in carrying else "They are not carrying anything."
                
                self.pub_action.publish(String(data=json.dumps({"action": "SAY_TEXT", "text": text})))
            
            elif self.state == "BONUS_VOICE_MOVE_TO_JUDGE2":
                self.state = "BONUS_SAYING_VOICE"
                voice = self.current_bonus_data.get("voice", {})
                allergy = voice.get("allergy", "unknown")
                nat = voice.get("nationality", "unknown")
                text = f"The first judge {allergy} and their nationality is {nat}."
                self.pub_action.publish(String(data=json.dumps({"action": "SAY_TEXT", "text": text})))

        elif msg.data == "BONUS_TEXT_DONE":
            if self.state == "BONUS_SAYING_VISION":
                # Now move back to Judge 1 for voice
                self.state = "BONUS_VOICE_MOVE_TO_JUDGE1"
                self.pub_action.publish(String(data=json.dumps({"action": "MOVE_TO_FACE_JUDGE_1"})))
            elif self.state == "BONUS_SAYING_VOICE":
                self.get_logger().info("BONUS PHASE COMPLETED!")
                self.state = "FINISHED"

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TaskPlannerNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()