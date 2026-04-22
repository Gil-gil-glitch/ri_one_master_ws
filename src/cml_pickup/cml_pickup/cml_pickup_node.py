#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import time

class CMLPickupNode(Node):
    def __init__(self):
        super().__init__('cml_pickup_node')
        
        self.state = "IDLE"
        self.target_bag_side = None
        self.bag_locked = False
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/commands/velocity', 10)
        self.pickup_status_pub = self.create_publisher(String, '/pickup_status', 10)
        
        # OpenManipulator-X Publishers (Update topic names if your namespace is different)
        self.arm_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        
        # Subscribers
        self.create_subscription(String, '/cml_state', self.state_callback, 10)
        self.create_subscription(String, '/target_bag', self.target_bag_callback, 10)
        self.create_subscription(String, '/bag_positions_3d', self.bag_positions_callback, 10)
        
        # Tuning parameters
        self.approach_distance = 0.45 # Stop 45cm away to switch to arm camera
        self.kp_linear = 0.5
        self.kp_angular = 1.0
        
        self.get_logger().info("Pickup node ready.")

    def state_callback(self, msg):
        self.state = msg.data

    def target_bag_callback(self, msg):
        self.target_bag_side = msg.data

    def bag_positions_callback(self, msg):
        if self.state != "PICKUP" or self.target_bag_side == "NONE":
            return
            
        if self.bag_locked:
            return

        bags = json.loads(msg.data)
        if len(bags) == 0:
            return
            
        bags.sort(key=lambda b: b['x']) 
        
        target_bag = None
        if self.target_bag_side == "LEFT" and len(bags) >= 1:
            target_bag = bags[0] 
        elif self.target_bag_side == "RIGHT" and len(bags) >= 2:
            target_bag = bags[-1] 
        elif self.target_bag_side == "RIGHT" and len(bags) == 1:
            target_bag = bags[0] 

        if target_bag:
            self.approach_bag(target_bag)

    def approach_bag(self, bag):
        twist = Twist()
        error_z = bag['z'] - self.approach_distance
        error_x = bag['x'] 
        
        if error_z > 0.05:
            twist.linear.x = min(self.kp_linear * error_z, 0.3)
            twist.angular.z = -self.kp_angular * error_x
            self.cmd_vel_pub.publish(twist)
        else:
            self.cmd_vel_pub.publish(Twist())
            self.bag_locked = True
            self.get_logger().info("RealSense Approach complete. Starting Hook Sequence.")
            self.execute_hook_sequence()

    def send_arm_pose(self, j1, j2, j3, j4, duration_sec):
        """Helper to send standard joint angles to OpenManipulator-X"""
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        point = JointTrajectoryPoint()
        point.positions = [j1, j2, j3, j4]
        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        
        msg.points.append(point)
        self.arm_pub.publish(msg)

    def send_gripper_pose(self, position, duration_sec):
        """Helper to open/close the gripper (-0.01 is fully closed, 0.01 is fully open)"""
        msg = JointTrajectory()
        msg.joint_names = ['gripper']
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        
        msg.points.append(point)
        self.gripper_pub.publish(msg)

    def execute_hook_sequence(self):
        # STEP 1: Lower Arm & Close Gripper (Hook formation)
        self.get_logger().info("Lowering arm and closing gripper to form hook...")
        
        # Close Gripper tight
        self.send_gripper_pose(-0.01, 1)
        
        # Move arm to a forward-reaching "hook" pose (You may need to tune these angles!)
        # J1: Base, J2: Shoulder, J3: Elbow, J4: Wrist
        self.send_arm_pose(0.0, 0.5, -0.5, -0.5, 2)
        time.sleep(2.5) 
        
        # STEP 2: Blind Drive Forward to slot the hook under the handle
        self.get_logger().info("Driving forward to hook handle...")
        twist = Twist()
        twist.linear.x = 0.15 
        self.cmd_vel_pub.publish(twist)
        time.sleep(1.5)
        self.cmd_vel_pub.publish(Twist()) # Stop
        time.sleep(0.5)
        
        # STEP 3: Lift the arm slightly
        self.get_logger().info("Lifting bag...")
        # Arching joint 2 (Shoulder) back to lift the bag off the ground
        self.send_arm_pose(0.0, -0.2, -0.3, -0.2, 2)
        time.sleep(2.5)
        
        # STEP 4: Tell Coordinator we are done
        msg = String()
        msg.data = "DONE"
        self.pickup_status_pub.publish(msg)
        
        self.bag_locked = False
        self.target_bag_side = "NONE"

def main(args=None):
    rclpy.init(args=args)
    # FIXED: Class name matches instantiation
    node = CMLPickupNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()