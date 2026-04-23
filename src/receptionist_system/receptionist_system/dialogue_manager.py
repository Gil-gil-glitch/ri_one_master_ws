import json
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time

class DialogueManager(Node):
    def __init__(self):
        super().__init__('dialogue_manager')

        # Subscribers / Publishers
        self.sub_action = self.create_subscription(String, '/task_action', self.action_cb, 10)
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)
        self.pub_status = self.create_publisher(String, '/action_status', 10) 
        
        # Action Clients
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

        # 座標データ（メモした数値に必要に応じて書き換えてください）
        self.poses = {
           'DOOR': {'x': -1.725, 'y': 0.86, 'w': -3.097},
           'HOST': {'x': -5.69, 'y': 2.73, 'w': 1.0},
           'SEAT': {'x': -3.53, 'y': 0.15, 'w': 1.0}
        }
        
        self.arm_poses = {
            'point_seat': [0.5, 0.4, 0.2, 0.0],
            'home': [0.0, 0.0, 0.0, 0.0]
        }
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_guest = {}
        self.record_door_pose()  

        self.get_logger().info("Dialogue Manager with Navigation started.")

    def record_door_pose(self):
        try:
            # Get the robot's current pose in the map frame
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time())
            self.poses['DOOR'] = {
                'x': t.transform.translation.x,
                'y': t.transform.translation.y,
                'w': t.transform.rotation.w # Also grab x, y, z for quaternion if needed
            }
            
            self.get_logger().info(f"DOOR position recorded dynamically.")
        except Exception as e:
            self.get_logger().warn(f"Could not record DOOR: {e}")
            
    def action_cb(self, msg):
        try:
            req = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("JSON Decode Error")
            return

        action = req.get("action")
        
        if action == "MOVE_TO_HOST":
            self.current_guest = req.get("data")
            self.say(f"Follow me, {self.current_guest['name']}. I will take you to Max.")
            # 移動命令を開始
            self.send_navigation_goal('HOST')
            
        elif action == "MOVE_TO_DOOR":
            self.say("I am going back to the entrance.")
            self.send_navigation_goal('DOOR')

    def send_navigation_goal(self, location_name):
        """Nav2へ移動命令を送信する"""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available')
            return

        pose_data = self.poses[location_name]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = pose_data['x']
        goal_msg.pose.pose.position.y = pose_data['y']
        goal_msg.pose.pose.orientation.w = pose_data.get('w', 1.0)


        self.get_logger().info(f"Navigating to {location_name}...")
        
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        

        # 目的地が受理されたか確認するコールバック
        send_goal_future.add_done_callback(lambda future: self.goal_response_callback(future, location_name))

    def goal_response_callback(self, future, location_name):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        get_result_future = goal_handle.get_result_async()
        # 到着（完了）したか確認するコールバック
        get_result_future.add_done_callback(lambda future: self.get_result_callback(future, location_name))

    def get_result_callback(self, future, location_name):
        """移動が完了した後の処理"""
        result = future.result().status
        if result == 4: # GoalStatus.STATUS_SUCCEEDED (ROS2 Humbleの標準)
            self.get_logger().info(f'Arrived at {location_name}!')
            
            if location_name == 'HOST':
                # ホストに到着したので紹介を開始
                self.introduce_guest()
            elif location_name == 'DOOR':
                # 入口に到着したのでPlannerに報告
                self.pub_status.publish(String(data="ARRIVED_AT_DOOR"))
        else:
            self.get_logger().warn(f'Navigation failed with status: {result}')

    def introduce_guest(self):
        """紹介と指差し動作"""
        self.say(f"Hello Max! I have brought {self.current_guest.get('name', 'someone')}.")
        self.say(f"Their favorite drink is {self.current_guest.get('drink', 'something')}.")
        
        # 指差し動作
        self.move_arm(self.arm_poses['point_seat'])
        self.say("There is an empty seat for you. Please take a seat.")
        
        # 紹介が終わったことをPlannerに報告
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