import json
import re

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class NLPNode(Node):
    def __init__(self):
        super().__init__('nlp_node')

        self.sub = self.create_subscription(
            String, '/speech_text', self.callback, 10)

        self.pub_intent = self.create_publisher(String, '/intent', 10)
        self.pub_entities = self.create_publisher(String, '/entities', 10)

        self.get_logger().info('NLP Node with Synonyms started')

        # ===== synonym辞書 =====
        self.synonyms = {
            "move": ["go", "walk", "navigate", "head"],
            "pick": ["grab", "take", "get"],
            "open": ["unlock"],
            "close": ["shut"],
            "turn_on": ["switch on", "activate"],
            "turn_off": ["switch off", "deactivate"],
        }

    # =====================
    def normalize_text(self, text):
        t = text.lower()

        # synonym置換
        for base, words in self.synonyms.items():
            for w in words:
                t = re.sub(rf"\b{w}\b", base, t)

        return t

    # =====================
    def classify_intent(self, text):

        if any(x in text for x in ['hello', 'hi', 'hey']):
            return 'greeting'

        if 'my name is' in text or re.search(r'\bi am [a-z]+', text):
            return 'introduce'

        if 'weather' in text:
            return 'ask_weather'

        if 'time' in text:
            return 'ask_time'

        if 'date' in text or 'day' in text:
            return 'ask_date'

        if 'where is' in text:
            return 'ask_location'

        if any(x in text for x in ['move', 'pick', 'open', 'close', 'turn_on', 'turn_off']):
            return 'request_action'

        return 'unknown'

    # =====================
    def extract_entities(self, text):

        entities = {}

        # name
        name = re.search(r'my name is ([A-Za-z]+)', text)
        if name:
            entities["name"] = name.group(1)

        # location
        for loc in ['lab', 'office', 'kitchen', 'room']:
            if loc in text:
                entities["location"] = loc

        # object
        for obj in ['pen', 'book', 'key', 'door']:
            if obj in text:
                entities["object"] = obj

        return entities

    # =====================
    def callback(self, msg):
        raw_text = msg.data
        text = self.normalize_text(raw_text)

        intent = self.classify_intent(text)
        entities = self.extract_entities(text)

        self.pub_intent.publish(String(data=intent))
        self.pub_entities.publish(String(data=json.dumps(entities)))

        self.get_logger().info(f"[INPUT] {raw_text}")
        self.get_logger().info(f"[NORMALIZED] {text}")
        self.get_logger().info(f"[RESULT] {intent}, {entities}")


def main():
    rclpy.init()
    node = NLPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()