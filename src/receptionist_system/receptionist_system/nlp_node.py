import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class NLPNode(Node):
    def __init__(self):
        super().__init__('nlp_node')
        
        # 購読: ASR(音声認識)からのテキスト
        self.sub_speech = self.create_subscription(String, '/speech_text', self.speech_cb, 10)
        # 購読: 競技開始等のトリガー
        self.sub_instruction = self.create_subscription(String, '/nlp_instruction', self.instruction_cb, 10)
        
        # 公開: 解析したゲスト情報 (TaskPlannerへ)
        self.pub_profile = self.create_publisher(String, '/person_profile', 10)
        # 公開: ロボットの発話指示 (TTSノードまたはログ用)
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)

        # 競技ルール用データ
        self.names = ['Adam', 'Axel', 'Chris', 'Hunter', 'Jack', 'Paris', 'Robin', 'Olivia', 'William', 'Max']
        self.drinks = ['orange juice', 'coke', 'soda', 'coffee', 'cocoa', 'lemonade', 'coconut milk', 'green tea', 'black tea', 'wine']

        # ステート管理
        self.current_asking = None  # "NAME", "DRINK", or None
        self.guest_data = {"name": None, "drink": None}

    def instruction_cb(self, msg):
        """外部から受付開始の合図を受けた時"""
        if msg.data == "START_GUEST_RECEPTION":
            self.get_logger().info("Reception started.")
            self.guest_data = {"name": None, "drink": None}
            self.ask_name()

    def ask_name(self):
        self.current_asking = "NAME"
        self.say("Hello! Welcome to the arena. May I have your name, please?")

    def ask_drink(self):
        self.current_asking = "DRINK"
        self.say(f"Thank you, {self.guest_data['name']}. What would you like to drink?")

    def say(self, text):
        """ロボットに喋らせる"""
        self.get_logger().info(f"ROBOT SAYS: {text}")
        msg = String()
        msg.data = text
        self.pub_tts.publish(msg)

    def speech_cb(self, msg):
        """音声入力を解析してステートを進める"""
        if self.current_asking is None:
            return

        text = msg.data.lower()
        self.get_logger().info(f"User said: {text}")

        if self.current_asking == "NAME":
            for name in self.names:
                if name.lower() in text:
                    self.guest_data["name"] = name
                    self.ask_drink()
                    return
            self.say("Sorry, I couldn't catch your name. Could you repeat it?")

        elif self.current_asking == "DRINK":
            for drink in self.drinks:
                if drink in text:
                    self.guest_data["drink"] = drink
                    self.finalize()
                    return
            self.say("Sorry, which drink did you say?")

    def finalize(self):
        """情報を確定させてTaskPlannerへ通知"""
        self.say(f"OK, I've got it. You are {self.guest_data['name']} and you like {self.guest_data['drink']}. Follow me!")
        
        res = String()
        res.data = json.dumps(self.guest_data)
        self.pub_profile.publish(res)
        self.current_asking = None

def main():
    rclpy.init()
    rclpy.spin(NLPNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()