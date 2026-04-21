import json
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
# アクション用にインポートを変更
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

class DialogueManager(Node):
    def __init__(self):
        super().__init__('dialogue_manager')

        self.sub_action = self.create_subscription(String, '/task_action', self.action_cb, 10)
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)
        self.pub_status = self.create_publisher(String, '/action_status', 10) 
        
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # --- 【修正】アーム用のクライアントをサービスからアクションに変更 ---
        self.arm_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/arm_controller/follow_joint_trajectory'
        )

        self.poses = {
            'DOOR': {'x': 0.0, 'y': 0.0, 'w': 1.0},
            'HOST': {'x': 2.5, 'y': 1.0, 'w': 1.0},
            'SEAT': {'x': 2.0, 'y': -1.0, 'w': 0.7} 
        }
        
        # 指差し動作の関節角度 [joint1, joint2, joint3, joint4]
        self.arm_poses = {
            'point_seat': [0.5, 0.4, 0.2, 0.0], # 席を指す
            'home': [0.0, 0.0, 0.0, 0.0]        # 基本姿勢
        }
        
        self.current_guest = {}

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
            # 本来は移動後に紹介ですが、デバッグ用に直接紹介へ
            self.introduce_guest() 
            
        elif action == "MOVE_TO_DOOR":
            self.say("I am going back to the entrance.")
            # ナビゲーションの完了を待たずに報告する場合の例
            self.pub_status.publish(String(data="ARRIVED_AT_DOOR"))

    def introduce_guest(self):
        """ホストに到着した時の紹介処理"""
        self.say(f"Hello Max! I have brought {self.current_guest.get('name', 'someone')}.")
        self.say(f"Their favorite drink is {self.current_guest.get('drink', 'something')}.")
        
        # --- 【修正】指差し動作 ---
        self.move_arm(self.arm_poses['point_seat'])
        self.say("There is an empty seat for you. Please take a seat.")
        
        # Plannerへ終了を報告
        self.pub_status.publish(String(data="COMPLETED_GUEST_MANAGEMENT"))

    def move_arm(self, pos):
        """【修正】アクションを使ってアームを動かすメソッド"""
        if not self.arm_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Arm Action Server not available')
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start.sec = 2 # 2秒かけて動く
        
        goal_msg.trajectory.points = [point]
        
        self.get_logger().info(f'Sending arm goal: {pos}')
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