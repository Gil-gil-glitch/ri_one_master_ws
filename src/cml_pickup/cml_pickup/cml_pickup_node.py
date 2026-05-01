import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

class CMLReleaseNode(Node):
    def __init__(self):
        super().__init__('cml_release_node')
        
        self.state = "IDLE"
        self.release_executed = False
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/pickup_status', 10)
        self.arm_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        
        # Subscribers
        self.create_subscription(String, '/cml_state', self.state_callback, 10)
        self.get_logger().info("Release/Drop node ready.")

    def state_callback(self, msg):
        self.state = msg.data
        if self.state == "DROP" and not self.release_executed:
            self.execute_drop_sequence()

    def send_arm_pose(self, j1, j2, j3, j4, duration_sec):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        point = JointTrajectoryPoint()
        point.positions = [j1, j2, j3, j4]
        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        msg.points.append(point)
        self.arm_pub.publish(msg)

    def send_gripper_pose(self, position, duration_sec):
        msg = JointTrajectory()
        msg.joint_names = ['gripper']
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        msg.points.append(point)
        self.gripper_pub.publish(msg)

    def execute_drop_sequence(self):
        self.release_executed = True
        self.get_logger().info("Executing Drop Sequence...")
        
        # 1. Lower arm slightly to prepare for release
        self.send_arm_pose(0.0, 0.6, -0.4, -0.4, 2)
        time.sleep(2.5)
        
        # 2. Open gripper fully
        self.send_gripper_pose(0.01, 1) # Assuming 0.01 is open
        time.sleep(1.5)
        
        # 3. Move arm back to home to clear the bag
        self.send_arm_pose(0.0, 0.0, 0.0, 0.0, 2)
        
        msg = String()
        msg.data = "RELEASED"
        self.status_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CMLReleaseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()