import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

class ReturnHomeNode(Node):
    def __init__(self):
        super().__init__('return_home_node')

        self.mode = "IDLE"
        self.home_pose = None
        self.goal_handle = None

        self.state_sub = self.create_subscription(String, '/cml_state', self.state_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info("Return Home node ready")

    def odom_callback(self, msg):
        if self.home_pose is None:
            self.home_pose = msg.pose.pose
            self.get_logger().info("Home recorded")

    def state_callback(self, msg):
        if msg.data == "RETURN" and self.mode != "RETURN":
            self.mode = "RETURN"
            self.send_goal()

        elif msg.data != "RETURN":
            self.mode = msg.data

    def send_goal(self):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 unavailable")
            return

        if self.home_pose is None:
            self.get_logger().error("No home pose")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'odom'
        goal_msg.pose.pose = self.home_pose

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Returning home...")
        self.goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Arrived home")

        self.mode = "IDLE"
        self.goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = ReturnHomeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()