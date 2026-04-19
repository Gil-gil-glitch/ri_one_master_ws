import unittest
import rclpy
from std_msgs.msg import String

from carry_my_luggage.cml_planner_node import CMLPlannerNode

class TestCMLPlannerNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = CMLPlannerNode()

    def tearDown(self):
        self.node.destroy_node()

    def test_initial_state(self):
        self.assertEqual(self.node.state, "START")

    def test_full_state_transition(self):
        # Create standard String messages
        msg_start = String()
        msg_start.data = "start the task"

        msg_gesture = String()
        msg_gesture.data = "left_bag"

        msg_grasp_success = String()
        msg_grasp_success.data = "SUCCESS"

        msg_reached = String()
        msg_reached.data = "we reached the car"

        msg_dropped = String()
        msg_dropped.data = "DROPPED"

        msg_nav_done = String()
        msg_nav_done.data = "REACHED_GOAL"

        # 1. START -> POINTING
        self.node.voice_cb(msg_start)
        self.assertEqual(self.node.state, "POINTING")

        # 2. POINTING -> GRASPING
        self.node.gesture_cb(msg_gesture)
        self.assertEqual(self.node.state, "GRASPING")

        # 3. GRASPING -> WAIT_FOLLOW -> FOLLOWING
        self.node.grasp_cb(msg_grasp_success)
        self.assertEqual(self.node.state, "FOLLOWING")

        # 4. FOLLOWING -> HANDOVER
        self.node.voice_cb(msg_reached)
        self.assertEqual(self.node.state, "HANDOVER")

        # 5. HANDOVER -> NAVIGATION_BACK
        self.node.grasp_cb(msg_dropped)
        self.assertEqual(self.node.state, "NAVIGATION_BACK")

        # 6. NAVIGATION_BACK -> DONE
        self.node.nav_cb(msg_nav_done)
        self.assertEqual(self.node.state, "DONE")

if __name__ == '__main__':
    unittest.main()
