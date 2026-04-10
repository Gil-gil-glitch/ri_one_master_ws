import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DialogueManager(Node):

    def __init__(self):
        super().__init__('dialogue_manager')

        self.sub_intent = self.create_subscription(
            String, '/intent', self.cb_intent, 10)

        self.sub_entities = self.create_subscription(
            String, '/entities', self.cb_entities, 10)

        self.pub_action = self.create_publisher(String, '/dialogue_action', 10)
        self.pub_profile = self.create_publisher(String, '/person_profile', 10)

        # ===== 状態 =====
        self.intent = None
        self.entities = {}

        # ===== ユーザ情報 =====
        self.profile = {}

        # ===== 会話状態 =====
        self.state = "idle"
        self.last_question = None

        # ===== sync flags — wait for both intent and entities =====
        self._intent_ready = False
        self._entities_ready = False

        self.get_logger().info("Dialogue Manager started")

    # =====================
    def cb_intent(self, msg):
        self.intent = msg.data
        self._intent_ready = True
        self._try_process()

    def cb_entities(self, msg):
        try:
            self.entities = json.loads(msg.data)
        except:
            self.entities = {}
        self._entities_ready = True
        self._try_process()

    def _try_process(self):
        if not (self._intent_ready and self._entities_ready):
            return
        # Reset flags for next utterance
        self._intent_ready = False
        self._entities_ready = False
        self.process()

    # =====================
    def process(self):

        if self.intent is None:
            return

        action = "none"

        # ===== Greeting =====
        if self.intent == "greeting":
            action = "say_hello"

        # ===== 名前登録 =====
        elif self.intent == "introduce":
            if "name" in self.entities:
                self.profile["name"] = self.entities["name"]
                action = f"store_name:{self.entities['name']}"
            else:
                action = "ask_name"
                self.state = "waiting_name"

        # ===== 名前待ち状態 =====
        elif self.state == "waiting_name":
            if "name" in self.entities:
                self.profile["name"] = self.entities["name"]
                action = f"store_name:{self.entities['name']}"
                self.state = "idle"

        # ===== Yes/No対応 =====
        elif self.intent == "unknown":
            if self.last_question == "confirm_move":
                if "yes" in self.entities or "yes" in str(self.entities):
                    action = "execute_move"
                else:
                    action = "cancel_move"

        # ===== 行動 =====
        elif self.intent == "request_action":
            if "location" in self.entities:
                action = f"move:{self.entities['location']}"
                self.last_question = "confirm_move"
            elif "object" in self.entities:
                action = f"pick:{self.entities['object']}"

        # ===== publish =====
        self.pub_action.publish(String(data=action))
        self.pub_profile.publish(String(data=json.dumps(self.profile)))

        self.get_logger().info(f"[ACTION] {action}")
        self.get_logger().info(f"[PROFILE] {self.profile}")

def main(args=None):
    rclpy.init(args=args)
    node = DialogueManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down dialogue manager")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()