import json
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

class DialogueManager(Node):
    def __init__(self):
        super().__init__('dialogue_manager')

        # === Publishers / Subscribers ===
        self.sub_action = self.create_subscription(String, '/task_action', self.action_cb, 10)
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)

        # === Nav2 Action Client ===
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # === 座標の設定 (競技環境に合わせて書き換えてください) ===
        # 例: ホスト(Max)が待っている座標
        self.host_pose = {
            'x': 2.5, 
            'y': 1.0, 
            'z': 0.0, 
            'w': 1.0
        }

        self.get_logger().info("Dialogue Manager with Nav2 integration started")

    def say(self, text):
        self.get_logger().info(f"ROBOT SAYS: {text}")
        msg = String()
        msg.data = text
        self.pub_tts.publish(msg)

    def action_cb(self, msg):
        try:
            data = json.loads(msg.data)
            action = data.get("action")
            info = data.get("data", {})

            if action == "MOVE_TO_HOST":
                self.start_navigation_to_host(info)
        except Exception as e:
            self.get_logger().error(f"Error in action_cb: {e}")

    def start_navigation_to_host(self, info):
        """移動を開始する"""
        name = info.get('name', 'guest')
        self.current_guest_info = info # 到着後に使うため保存

        self.say(f"Okay {name}, please follow me. I will take you to Max.")

        # Nav2のサーバーが立ち上がっているか確認
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            self.say("I am sorry, I cannot move right now.")
            return

        # ゴールの作成
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = self.host_pose['x']
        goal_msg.pose.pose.position.y = self.host_pose['y']
        goal_msg.pose.pose.orientation.z = self.host_pose['z']
        goal_msg.pose.pose.orientation.w = self.host_pose['w']

        self.get_logger().info(f"Sending goal to Nav2: x={self.host_pose['x']}, y={self.host_pose['y']}")
        
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected :(")
            return

        self.get_logger().info("Goal accepted :)")
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """目的地に到着した時に呼ばれる"""
        result = future.result().status
        if result == 4: # GoalStatus.STATUS_SUCCEEDED (4)
            self.get_logger().info("Arrived at host location!")
            self.introduce_guest()
        else:
            self.get_logger().warn(f"Navigation failed with status: {result}")
            self.say("I am lost, but I will try to introduce you anyway.")
            self.introduce_guest()

    def introduce_guest(self):
        """ホストへの紹介を実行"""
        name = self.current_guest_info.get('name', 'guest')
        drink = self.current_guest_info.get('drink', 'something')
        
        self.say(f"Hello Max! I have brought a guest for you.")
        # 少し間を置いて情報を伝える
        self.say(f"This is {name}, and their favorite drink is {drink}.")
        self.say("Please enjoy your time.")

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