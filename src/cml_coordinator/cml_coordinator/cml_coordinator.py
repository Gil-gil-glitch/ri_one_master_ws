import rclpy
from rclpy.node import Node
from std_msgs.msg import String
 
 
class CMLCoordinator(Node):
    """
    State machine:  READY_TO_LOAD  ->  FOLLOW  ->  DROP  ->  IDLE
 
    Voice commands (published by cml_asr.py on /voice_imperatives):
        "following"  ->  transition to FOLLOW
        "returning"  ->  transition to DROP   ← FIX: was checking "stop", ASR sends "returning"
 
    Pickup status (published by cml_pickup_node.py on /pickup_status):
        "RELEASED"   ->  transition to IDLE
    """
 
    VALID_TRANSITIONS = {
        "READY_TO_LOAD": ["FOLLOW"],
        "FOLLOW":        ["DROP"],
        "DROP":          ["IDLE"],
        "IDLE":          [],
    }
 
    def __init__(self):
        super().__init__('cml_coordinator')
 
        self.state = "READY_TO_LOAD"
        self._last_published_state = None   # Only publish on actual changes
 
        self.state_pub = self.create_publisher(String, '/cml_state', 10)
        self.pub_tts   = self.create_publisher(String, '/robot_speech', 10)
 
        self.command_sub      = self.create_subscription(String, '/voice_imperatives', self.command_callback, 10)
        self.pickup_status_sub = self.create_subscription(String, '/pickup_status',    self.pickup_status_callback, 10)
 
        # Startup prompt after 2 s
        self.startup_timer = self.create_timer(2.0, self.startup_routine)
        self.prompt_given  = False
 
        # Heartbeat: re-publish current state every 2 s so late-joining nodes catch up.
        # Using 2 s (not 1 s) to reduce the message flood that confused cml_pickup_node.
        self.heartbeat_timer = self.create_timer(2.0, self.publish_state)
 
        self.get_logger().info("CML Coordinator started. State: READY_TO_LOAD")
 
    # ------------------------------------------------------------------ #
    #  Startup                                                             #
    # ------------------------------------------------------------------ #
    def startup_routine(self):
        if not self.prompt_given:
            self.say("Please give me the bag and say 'follow me' when you are ready.")
            self.prompt_given = True
            self.startup_timer.cancel()
            self.publish_state()   # Broadcast initial state immediately
 
    # ------------------------------------------------------------------ #
    #  Voice command handler                                               #
    # ------------------------------------------------------------------ #
    def command_callback(self, msg):
        command = msg.data.lower()
        self.get_logger().info(f"Voice command received: '{command}'  (current state: {self.state})")
 
        # FIX #1: ASR publishes "following" and "returning" — match those exact strings.
        if "follow" in command and self.state == "READY_TO_LOAD":
            self._transition_to("FOLLOW", "Okay, I will follow you.")
 
        elif "return" in command and self.state == "FOLLOW":
            # "returning" sent by ASR on "thank you" / "thanks"
            self._transition_to("DROP", "Thank you! Dropping the bag now.")
 
        else:
            self.get_logger().warn(
                f"Command '{command}' ignored — not valid in state '{self.state}'"
            )
 
    # ------------------------------------------------------------------ #
    #  Pickup / release status                                            #
    # ------------------------------------------------------------------ #
    def pickup_status_callback(self, msg):
        if msg.data == "RELEASED" and self.state == "DROP":
            self._transition_to("IDLE", "Bag released. Mission complete.")
 
    # ------------------------------------------------------------------ #
    #  State management                                                    #
    # ------------------------------------------------------------------ #
    def _transition_to(self, new_state: str, speech: str = ""):
        allowed = self.VALID_TRANSITIONS.get(self.state, [])
        if new_state not in allowed:
            self.get_logger().error(
                f"Illegal transition {self.state} -> {new_state}. Ignored."
            )
            return
        self.get_logger().info(f"State: {self.state} -> {new_state}")
        self.state = new_state
        if speech:
            self.say(speech)
        self.publish_state()   # Publish immediately on change
 
    def publish_state(self):
        """Publish current state. Called on change and by heartbeat timer."""
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)
        if self.state != self._last_published_state:
            self.get_logger().info(f"Published state: {self.state}")
            self._last_published_state = self.state
 
    def say(self, text: str):
        msg = String()
        msg.data = text
        self.pub_tts.publish(msg)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = CMLCoordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
