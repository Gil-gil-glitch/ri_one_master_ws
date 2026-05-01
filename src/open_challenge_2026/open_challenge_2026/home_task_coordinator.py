#
##  Home Task Coordinator
#
#  Subscribes to /home_tasks and runs a state machine for the grocery_shopping task:
#
#    IDLE
#      └─► APPROACH      publish "approach_person" → /open_challenge_state
#            └─► WAIT_FOR_GESTURE   speak prompt, wait for "thumbsup" on /gesture
#                  └─► FOLLOW       lock target, set /cml_state "FOLLOW" for 5 s
#                        └─► IDLE   release lock, done
#
#  Interfaces used:
#    Sub:  /home_tasks          (std_msgs/String)  — triggers task
#    Sub:  /gesture             (std_msgs/String)  — "thumbsup" advances state
#    Pub:  /open_challenge_state (std_msgs/String) — "approach_person"
#    Pub:  /cml_state           (std_msgs/String)  — "FOLLOW" / "IDLE"
#    Pub:  /voice_imperatives   (std_msgs/String)  — "following" / "returning"
#    Pub:  /robot_speech        (std_msgs/String)  — TTS text
#

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# How long (seconds) the approach phase runs before transitioning to gesture wait.
# Set this long enough for the approaching_person_node to finish its drive.
APPROACH_DURATION = 10.0

# How long (seconds) the robot follows before stopping.
FOLLOW_DURATION = 20.0

# How long (seconds) to wait between repeated TTS prompts during WAIT_FOR_GESTURE.
REPROMPT_INTERVAL = 6.0


class HomeTaskCoordinator(Node):

    STATES = ["IDLE", "APPROACH", "WAIT_FOR_GESTURE", "FOLLOW"]

    def __init__(self):
        super().__init__('home_task_coordinator')

        # ── Subscribers ──────────────────────────────────────────────────────
        self.task_sub = self.create_subscription(
            String, '/home_tasks', self.task_callback, 10)

        self.gesture_sub = self.create_subscription(
            String, '/gesture', self.gesture_callback, 10)

        # ── Publishers ───────────────────────────────────────────────────────
        self.open_challenge_pub = self.create_publisher(
            String, '/open_challenge_state', 10)

        self.cml_state_pub = self.create_publisher(
            String, '/cml_state', 10)

        self.voice_imperatives_pub = self.create_publisher(
            String, '/voice_imperatives', 10)

        self.speech_pub = self.create_publisher(
            String, '/robot_speech', 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.state = "IDLE"
        self._phase_timer = None      # active one-shot timer for phase transitions
        self._reprompt_timer = None   # repeating timer for TTS nudges

        self.get_logger().info("Home Task Coordinator ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _cancel_timer(self, timer):
        """Safely cancel a timer that may already be None or expired."""
        if timer is not None:
            timer.cancel()
        return None

    def _publish(self, publisher, text: str):
        msg = String()
        msg.data = text
        publisher.publish(msg)

    def _speak(self, text: str):
        self.get_logger().info(f'TTS: "{text}"')
        self._publish(self.speech_pub, text)

    def _set_state(self, new_state: str):
        self.get_logger().info(f"State: {self.state} → {new_state}")
        self.state = new_state

    # ─────────────────────────────────────────────────────────────────────────
    # External triggers
    # ─────────────────────────────────────────────────────────────────────────

    def task_callback(self, msg: String):
        if msg.data != "grocery_shopping":
            self.get_logger().warn(f"Unknown task: '{msg.data}' — ignoring.")
            return

        if self.state != "IDLE":
            self.get_logger().warn(
                f"Received grocery_shopping task but state is {self.state}. Ignoring.")
            return

        self.get_logger().info("grocery_shopping task received — starting approach.")
        self._enter_approach()

    def gesture_callback(self, msg: String):
        if msg.data == "thumbsup" and self.state == "WAIT_FOR_GESTURE":
            self.get_logger().info("Thumbs-up detected — transitioning to FOLLOW.")
            self._enter_follow()

    # ─────────────────────────────────────────────────────────────────────────
    # State entries
    # ─────────────────────────────────────────────────────────────────────────

    def _enter_approach(self):
        """Tell the approaching_person_node to drive forward, then wait."""
        self._set_state("APPROACH")
        self._publish(self.open_challenge_pub, "approach_person")
        self._speak("I am moving to the door. Please load your groceries on my back.")

        # After APPROACH_DURATION seconds, move to gesture wait regardless.
        self._phase_timer = self._cancel_timer(self._phase_timer)
        self._phase_timer = self.create_timer(APPROACH_DURATION, self._approach_done)

    def _approach_done(self):
        """Called once APPROACH_DURATION has elapsed."""
        self._phase_timer = self._cancel_timer(self._phase_timer)

        if self.state != "APPROACH":
            return  # state changed from outside (shouldn't happen)

        self._enter_wait_for_gesture()

    def _enter_wait_for_gesture(self):
        """Speak prompt and wait indefinitely for a thumbs-up gesture."""
        self._set_state("WAIT_FOR_GESTURE")
        self._speak("Please give me a thumbs up when you are ready and I will follow you.")

        # Reprompt the user periodically so they know we are still waiting.
        self._reprompt_timer = self._cancel_timer(self._reprompt_timer)
        self._reprompt_timer = self.create_timer(REPROMPT_INTERVAL, self._reprompt)

    def _reprompt(self):
        if self.state == "WAIT_FOR_GESTURE":
            self._speak("Waiting for your thumbs up.")

    def _enter_follow(self):
        """Lock onto person via person_targetting_node and activate follow_node."""
        self._reprompt_timer = self._cancel_timer(self._reprompt_timer)
        self._set_state("FOLLOW")

        # Tell person_targetting_node to capture and lock onto the current person.
        self._publish(self.voice_imperatives_pub, "following")

        # Small delay so the targetting node has a couple of frames to acquire,
        # then tell follow_node to start.  We use a one-shot timer to avoid
        # blocking the executor.
        self._phase_timer = self._cancel_timer(self._phase_timer)
        self._phase_timer = self.create_timer(0.5, self._start_follow_node)

    def _start_follow_node(self):
        """Activate follow_node and schedule the stop."""
        self._phase_timer = self._cancel_timer(self._phase_timer)

        if self.state != "FOLLOW":
            return

        self._publish(self.cml_state_pub, "FOLLOW")
        self._speak("I will follow you now. Lead the way!")
        self.get_logger().info(f"Following for {FOLLOW_DURATION} seconds.")

        self._phase_timer = self.create_timer(FOLLOW_DURATION, self._follow_done)

    def _follow_done(self):
        """Stop following and return to IDLE."""
        self._phase_timer = self._cancel_timer(self._phase_timer)

        if self.state != "FOLLOW":
            return

        # Release person_targetting_node lock.
        self._publish(self.voice_imperatives_pub, "returning")

        # Stop follow_node.
        self._publish(self.cml_state_pub, "IDLE")

        self._speak("Thank you! Task complete. I will stop here.")
        self._set_state("IDLE")
        self.get_logger().info("Grocery shopping task complete.")


def main(args=None):
    rclpy.init(args=args)
    node = HomeTaskCoordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()