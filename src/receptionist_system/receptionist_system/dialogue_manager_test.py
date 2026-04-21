import json
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from open_manipulator_msgs.srv import SetJointPosition # アーム用

class DialogueManager(Node):
    def __init__(self):
        super().__init__('dialogue_manager')

        self.sub_action = self.create_subscription(String, '/task_action', self.action_cb, 10)
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)
        self.pub_status = self.create_publisher(String, '/action_status', 10) # Plannerへの報告用
        
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.arm_client = self.create_client(SetJointPosition, 'goal_joint_space_path')

        # 重要: RVizで調べた座標に書き換えてください！
        self.poses = {
            'DOOR': {'x': 0.0, 'y': 0.0, 'w': 1.0},
            'HOST': {'x': 2.5, 'y': 1.0, 'w': 1.0},
            'SEAT': {'x': 2.0, 'y': -1.0, 'w': 0.7} # 空席の場所
        }
        self.current_guest = {}

    def action_cb(self, msg):
        req = json.loads(msg.data)
        action = req.get("action")
        
        if action == "MOVE_TO_HOST":
            self.current_guest = req.get("data")
            self.say(f"Follow me, {self.current_guest['name']}. I will take you to Max.")
            self.send_nav_goal(self.poses['HOST'], callback=self.on_arrived_at_host)
            
        elif action == "MOVE_TO_DOOR":
            self.say("I am going back to the entrance.")
            self.send_nav_goal(self.poses['DOOR'], callback=self.on_arrived_at_door)

    def send_nav_goal(self, pose_data, callback):
        # goal_msg = NavigateToPose.Goal()
        # goal_msg.pose.header.frame_id = "map"
        # goal_msg.pose.pose.position.x = pose_data['x']
        # goal_msg.pose.pose.position.y = pose_data['y']
        # goal_msg.pose.pose.orientation.w = pose_data['w']
        
        # self.nav_client.wait_for_server()
        # future = self.nav_client.send_goal_async(goal_msg)
        # future.add_done_callback(lambda f: f.result().get_result_async().add_done_callback(callback))

        
        # 修正イメージ
        self.get_logger().info("Navigation skipped. Executing next task...")
        # 本来のアクション呼び出しをコメントアウトし、
        # 直接「紹介フェーズ」へ移行する関数を呼び出す
        self.introduce_guest()

    def on_arrived_at_host(self, future):
        """ホストに到着した時の処理"""
        self.say(f"Hello Max! I have brought {self.current_guest['name']}.")
        self.say(f"Their favorite drink is {self.current_guest['drink']}.")
        
        # 指差し動作 (アーム)
        self.move_arm([0.5, 0.0, 0.0, 0.0]) # 席の方向へ
        self.say("There is an empty seat for you. Please take a seat.")
        
        # Plannerへ終了を報告
        self.pub_status.publish(String(data="COMPLETED_GUEST_MANAGEMENT"))

    def on_arrived_at_door(self, future):
        self.pub_status.publish(String(data="ARRIVED_AT_DOOR"))

    def move_arm(self, pos):
        if self.arm_client.wait_for_service(timeout_sec=1.0):
            req = SetJointPosition.Request()
            req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4']
            req.joint_position.position = pos
            self.arm_client.call_async(req)

    def say(self, text):
        self.pub_tts.publish(String(data=text))

def main():
    rclpy.init()
    rclpy.spin(DialogueManager())
    rclpy.shutdown()