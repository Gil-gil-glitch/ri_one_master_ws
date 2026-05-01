import json
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import String

class NLPNode(Node):
    def __init__(self):
        super().__init__('nlp_node')

        # Subscribers
        self.sub_speech = self.create_subscription(String, '/speech_text', self.speech_cb, 10)
        self.sub_instruction = self.create_subscription(String, '/nlp_instruction', self.instruction_cb, 10)

        # Publishers
        self.pub_profile = self.create_publisher(String, '/person_profile', 10)
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)

        self.names = ['axel', 'chris', 'hunter', 'jack', 'paris', 'robin', 'olivia', 'william', 'max']
        self.drinks = ['tropical juice', 'coke', 'soda', 'coffee', 'calpis', 'water', 'milk', 'green tea', 'wine']
        self.countries = ['japan', 'malaysia', 'china', 'chinese', 'hong kong', 'indonesia', 'korean']

        self.current_asking = None
        self.guest_data = {"name": None, "drink": None, "allergy": None, "nationality": None}

        # Timeout management
        self.last_ask_time = None
        self.timeout_sec = 10.0
        self.has_asked_repeat = False
        self.timer = self.create_timer(1.0, self.check_timeout)

    def check_timeout(self):
        if self.current_asking is not None and self.last_ask_time is not None:
            elapsed = time.time() - self.last_ask_time
            if elapsed >= self.timeout_sec and not self.has_asked_repeat:
                self.get_logger().warn(f"Timeout ({self.timeout_sec}s) reached. Asking to repeat.")
                self.has_asked_repeat = True
                self.say("Sorry, I couldn't hear you. Could you say that again or speak louder, please?")
                # Reset time so it doesn't repeat immediately, but we only repeat once per question phase
                self.last_ask_time = time.time()

    def instruction_cb(self, msg):
        if msg.data == "START_GUEST_RECEPTION":
            self.get_logger().info("Reception started.")
            self.guest_data = {"name": None, "drink": None, "allergy": None, "nationality": None}
            self.ask_name()
        elif msg.data == "START_BONUS_VOICE":
            self.get_logger().info("Bonus voice phase started.")
            self.ask_allergy()

    def ask_name(self):
        self.current_asking = "NAME"
        self.has_asked_repeat = False
        self.say("hello welcome to the arena may i please have your name")

    def ask_drink(self):
        self.current_asking = "DRINK"
        self.has_asked_repeat = False
        self.say(f"Thank you, {self.guest_data['name']}. What would you like to drink?")

    def ask_allergy(self):
        self.current_asking = "ALLERGY"
        self.has_asked_repeat = False
        self.say("May I ask if you have any food allergies?")

    def ask_nationality(self):
        self.current_asking = "NATIONALITY"
        self.has_asked_repeat = False
        self.say("And what is your nationality?")

    def say(self, text):
        self.get_logger().info(f"ROBOT SAYS: {text}")
        self.pub_tts.publish(String(data=text))
        self.last_ask_time = time.time()

    def speech_cb(self, msg):
        if self.current_asking is None:
            return

        text = msg.data.strip().lower()
        if not text:
            return
            
        self.get_logger().info(f"User said: {text}")
        # Reset timeout on any valid speech received
        self.last_ask_time = time.time()

        if self.current_asking == "NAME":
            for name in self.names:
                if name in text:
                    self.guest_data["name"] = name.capitalize()
                    self.get_logger().info(f"Name recognized: {name}")
                    self.ask_drink()
                    return

        elif self.current_asking == "DRINK":
            for drink in self.drinks:
                if drink in text:
                    self.guest_data["drink"] = drink
                    self.get_logger().info(f"Drink recognized: {drink}")
                    self.finalize_reception()
                    return

        elif self.current_asking == "ALLERGY":
            if "yes" in text or "have" in text:
                self.guest_data["allergy"] = "has allergies"
                self.ask_nationality()
            elif "no" in text or "don't" in text:
                self.guest_data["allergy"] = "no allergies"
                self.ask_nationality()

        elif self.current_asking == "NATIONALITY":
            for country in self.countries:
                if country in text:
                    self.guest_data["nationality"] = country.capitalize()
                    self.finalize_bonus()
                    return

    def finalize_reception(self):
        name = self.guest_data["name"]
        drink = self.guest_data["drink"]
        self.current_asking = None
        self.last_ask_time = None
        self.say(f"OK, I've got it. You are {name} and you like {drink}.")
        self.pub_profile.publish(String(data=json.dumps(self.guest_data)))

    def finalize_bonus(self):
        self.current_asking = None
        self.last_ask_time = None
        allergy = self.guest_data["allergy"]
        nat = self.guest_data["nationality"]
        self.get_logger().info(f"Bonus Voice result: {allergy}, {nat}")
        self.pub_profile.publish(String(data=json.dumps({"type": "BONUS_VOICE_DONE", "allergy": allergy, "nationality": nat})))

def main():
    rclpy.init()
    rclpy.spin(NLPNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()