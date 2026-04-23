import json
import rclpy
import math
import numpy as np
import tf2_geometry_msgs

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped as MsgPointStamped, PoseStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge

# Native Nav2 Action
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

class DialogueManager(Node):
    def __init__(self):
        super().__init__('dialogue_manager')

        # Subscribers 
        self.sub_action = self.create_subscription(String, '/task_action', self.action_cb, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_cb, 10)
        self.create_subscription(String, '/receptionist/detections', self.detections_cb, 10)

        # Publishers
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)
        self.pub_status = self.create_publisher(String, '/action_status', 10) 
        
        # Action Clients
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        door_quat = self.euler_yaw_to_quat(-3.097)  

        self.poses = {
           'DOOR': {'x': -1.725, 'y': 0.86, 'z': door_quat['z'], 'w': door_quat['w']},
           'HOST': {  'x': 0.5006077885627747, 'y': 0.05317028984427452, 'z': 0.002471923828125, 'w': 1.0 },
           'SEAT': {'x': -3.53, 'y': 0.15, 'z': 0.0, 'w': 1.0}
        }   
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.latest_depth = None
        self.camera_info = None
        self.latest_detections = None

        self.current_guest = {}
        
        # Timer for dynamic door pose
        self.tf_timer = self.create_timer(2.0, self.record_door_pose)  

        self.get_logger().info("Dialogue Manager initialized. Waiting for commands on /task_action...")

    def record_door_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time())
            self.poses['DOOR'] = {
                'x': t.transform.translation.x,
                'y': t.transform.translation.y,
                'z': t.transform.rotation.z, 
                'w': t.transform.rotation.w
            }
            self.get_logger().info(f"DOOR position recorded dynamically.")
            self.tf_timer.cancel()
        except Exception as e:
            pass # Keep trying silently until TF map->base_link becomes available

    def euler_yaw_to_quat(self, yaw):
        return {'x': 0.0, 'y': 0.0, 'z': math.sin(yaw / 2.0), 'w': math.cos(yaw / 2.0)}

    # ======== Callbacks ========
    def depth_cb(self, msg):
        try: self.latest_depth = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        except Exception: pass

    def camera_info_cb(self, msg):
        self.camera_info = msg

    def detections_cb(self, msg):
        try: self.latest_detections = json.loads(msg.data)
        except Exception: pass

    def action_cb(self, msg):
        try:
            req = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("JSON Decode Error")
            return

        action = req.get("action")
        
        if action == "MOVE_TO_HOST":
            self.current_guest = req.get("data", {})
            guest_name = self.current_guest.get('name', 'guest')
            self.say(f"Follow me, {guest_name}. I will take you to Max.")
            self.send_navigation_goal('HOST')
            
        elif action == "MOVE_TO_DOOR":
            self.say("I am going back to the entrance.")
            self.send_navigation_goal('DOOR')

    # ======== Navigation Logic ========
    def send_navigation_goal(self, location_name):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("NavigateToPose action server not available!")
            return

        pose_data = self.poses[location_name]

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = pose_data['x']
        goal_msg.pose.pose.position.y = pose_data['y']
        goal_msg.pose.pose.orientation.z = pose_data.get('z', 0.0)
        goal_msg.pose.pose.orientation.w = pose_data.get('w', 1.0)

        self.get_logger().info(f"Sending goal to {location_name}...")
        
        # Async goal sending
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(lambda future: self.goal_response_callback(future, location_name))

    def goal_response_callback(self, future, location_name):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f'Goal to {location_name} was rejected by Nav2!')
            return

        self.get_logger().info(f'Goal accepted, navigating to {location_name}...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(lambda future: self.get_result_callback(future, location_name))

    def get_result_callback(self, future, location_name):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Successfully arrived at {location_name}!')
            if location_name == 'HOST':
                self.introduce_guest()
            elif location_name == 'DOOR':
                self.pub_status.publish(String(data="ARRIVED_AT_DOOR"))
        else:
            self.get_logger().warn(f'Navigation to {location_name} failed with status code: {status}')

    # ======== Arm & Speech Logic ========
    def introduce_guest(self):
        self.say(f"Hello Max! I have brought {self.current_guest.get('name', 'someone')}.")
        self.say(f"Their favorite drink is {self.current_guest.get('drink', 'something')}.")
        
        # Safely point to seat
        self.move_arm([0.0, 0.4, 0.2, 0.0]) # Simplified for testing, you can add dynamic logic back once it moves
        self.say("There is an empty seat for you. Please take a seat.")
        self.pub_status.publish(String(data="COMPLETED_GUEST_MANAGEMENT"))

    def move_arm(self, pos):
        if not self.arm_client.wait_for_server(timeout_sec=1.0):
            return
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start.sec = 2
        goal_msg.trajectory.points = [point]
        self.arm_client.send_goal_async(goal_msg)

    def say(self, text):
        self.pub_tts.publish(String(data=text))

def main(args=None):
    rclpy.init(args=args)
    node = DialogueManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()