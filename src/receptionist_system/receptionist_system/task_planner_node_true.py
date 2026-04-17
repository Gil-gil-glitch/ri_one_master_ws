import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # === Subscribers (購読) ===
        # 1. Visionノードからの検知結果（距離データ付き）を購読
        self.sub_vision = self.create_subscription(
            String, '/receptionist/detections', self.vision_cb, 10)
        
        # 2. NLPノードから確定したプロフィールを購読
        self.sub_profile = self.create_subscription(
            String, '/person_profile', self.profile_cb, 10)

        # === Publishers (公開) ===
        # 1. NLPノードへ「受付開始」の指示を送る
        self.pub_nlp_trigger = self.create_publisher(String, '/nlp_instruction', 10)
        
        # 2. DialogueManagerへの行動指示（移動や停止）
        self.pub_action = self.create_publisher(String, '/task_action', 10)

        # === 内部状態の管理 (ステートマシン) ===
        # WAITING: 人を探している
        # APPROACHING: 人を見つけ、目標距離まで移動中
        # TALKING: 到着して対話中
        # DONE: 全工程終了
        self.state = "WAITING"
        
        # 近づくのを止める距離（メートル）
        self.target_arrival_dist = 1.2 

        self.get_logger().info("Task Planner (Auto-Approach mode) started.")

    def vision_cb(self, msg):
        """Visionノードから届くデータに基づいて状態を遷移させる"""
        try:
            data = json.loads(msg.data)
            status = data.get("status")
            people = data.get("people", [])

            # 状態1: 待機中に人を発見
            if self.state == "WAITING" and status == "person_detected":
                self.get_logger().info("Person detected! Initializing approach...")
                self.send_action("APPROACH_GUEST") # DialogueManagerに移動を指示
                self.state = "APPROACHING"

            # 状態2: 接近中。距離をチェックして到着判定
            elif self.state == "APPROACHING":
                if len(people) > 0:
                    # 最も近い人の距離を取得
                    current_dist = people[0].get('distance', 99.9)
                    
                    self.get_logger().info(f"Approaching... Current dist: {current_dist:.2f}m", once=True)

                    # 距離が 0.1m以上(エラー除外) かつ 目標距離(1.2m)以下になったら停止
                    if 0.1 < current_dist <= self.target_arrival_dist:
                        self.get_logger().info("Target distance reached. Stopping and starting dialogue.")
                        
                        # 停止と対話開始の指示
                        self.send_action("STOP_AND_GREET")
                        self.trigger_nlp() 
                        
                        self.state = "TALKING"
                else:
                    # 人を見失った場合の処理（必要に応じて）
                    self.get_logger().warn("Lost sight of person during approach.")

        except Exception as e:
            self.get_logger().error(f"Error in vision_cb: {e}")

    def profile_cb(self, msg):
        """対話が完了し、プロフィールが届いた時の処理"""
        if self.state != "TALKING":
            return

        try:
            profile_data = json.loads(msg.data)
            name = profile_data.get("name")
            drink = profile_data.get("drink")

            if name and drink:
                self.get_logger().info(f"Reception complete for {name}. Moving to Host.")
                
                # DialogueManagerにホストへの案内指示を送る
                instruction = {
                    "action": "MOVE_TO_HOST",
                    "data": {"name": name, "drink": drink}
                }
                msg_out = String()
                msg_out.data = json.dumps(instruction)
                self.pub_action.publish(msg_out)
                
                self.state = "DONE"

        except Exception as e:
            self.get_logger().error(f"Error in profile_cb: {e}")

    def send_action(self, action_name):
        """DialogueManagerへ指示を送信"""
        msg = String()
        msg.data = json.dumps({"action": action_name})
        self.pub_action.publish(msg)

    def trigger_nlp(self):
        """NLPノードへ対話開始を指示"""
        msg = String(data="START_GUEST_RECEPTION")
        self.pub_nlp_trigger.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()