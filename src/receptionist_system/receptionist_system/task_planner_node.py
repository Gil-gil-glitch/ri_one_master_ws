import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # === Subscribers (購読) ===
        # 1. Visionノードからの検知結果を購読
        self.sub_vision = self.create_subscription(
            String, '/receptionist/detections', self.vision_cb, 10)
        
        # 2. NLPノードから確定したプロフィールを購読
        self.sub_profile = self.create_subscription(
            String, '/person_profile', self.profile_cb, 10)

        # === Publishers (公開) ===
        # 1. NLPノードへ「受付開始」の指示を送る
        self.pub_nlp_trigger = self.create_publisher(String, '/nlp_instruction', 10)
        
        # 2. DialogueManagerへの行動指示
        self.pub_action = self.create_publisher(String, '/task_action', 10)

        # 状態管理変数
        self.last_vision_status = "searching"
        self.is_reception_active = False # 二重に挨拶しないためのフラグ

        self.get_logger().info("Task Planner Node (Vision-NLP Bridge) started.")

    def vision_cb(self, msg):
        """Visionノードから「人が来た」という情報を受け取った時の処理"""
        try:
            data = json.loads(msg.data)
            current_status = data.get("status") # "guest_arrived" か "searching"

            # 状態の変化をチェック: 探索中 -> ゲスト到着
            if current_status == "guest_arrived" and self.last_vision_status == "searching":
                if not self.is_reception_active:
                    self.get_logger().info("Guest detected! Sending trigger to NLP node...")
                    
                    # NLPノードに受付開始を指示
                    trigger_msg = String()
                    trigger_msg.data = "START_GUEST_RECEPTION"
                    self.pub_nlp_trigger.publish(trigger_msg)
                    
                    self.is_reception_active = True # 受付モードに移行

            self.last_vision_status = current_status
        except Exception as e:
            self.get_logger().error(f"Error in vision_cb: {e}")

    def profile_cb(self, msg):
        """NLPノードから名前と飲み物が確定して届いた時の処理"""
        try:
            profile_data = json.loads(msg.data)
            name = profile_data.get("name")
            drink = profile_data.get("drink")

            if name and drink:
                self.get_logger().info(f"Full profile received for {name}. Telling Dialogue Manager to move.")
                
                # 移動指示を作成
                instruction = {
                    "action": "MOVE_TO_HOST",
                    "data": {"name": name, "drink": drink}
                }
                msg_out = String()
                msg_out.data = json.dumps(instruction)
                self.pub_action.publish(msg_out)
                
                # 受付が完了したのでフラグを戻す（次の人のために）
                self.is_reception_active = False 

        except Exception as e:
            self.get_logger().error(f"Error in profile_cb: {e}")

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