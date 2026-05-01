import json
import rclpy
import time
import math
from rclpy.node import Node
from std_msgs.msg import String

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # --- 現場調整が必要なパラメータ ---
        # ホスト（Max）が立っている座標 (DialogueManagerの poses['HOST'] と合わせる)
        self.HOST_X = 0.50
        self.HOST_Y = 0.05
        self.HOST_IGNORE_RADIUS = 1.0  # ホストから1m以内にいる人はゲストとみなさない
        # ------------------------------

        # Subscribers
        self.sub_vision = self.create_subscription(String, '/receptionist/detections', self.vision_cb, 10)
        self.sub_profile = self.create_subscription(String, '/person_profile', self.profile_cb, 10)
        self.sub_action_status = self.create_subscription(String, '/action_status', self.action_status_cb, 10)

        # Publishers
        self.pub_nlp_trigger = self.create_publisher(String, '/nlp_instruction', 10)
        self.pub_action = self.create_publisher(String, '/task_action', 10)

        # 状態管理
        # WAITING_FOR_GUEST, APPROACHING_GUEST, RECEPTION, GOING_TO_HOST, RETURNING_TO_DOOR
        self.state = "WAITING_FOR_GUEST" 
        self.guest_count = 0
        self.last_vision_status = "searching"
        
        # クールタイム用
        self.last_reception_time = 0
        self.cooldown_period = 5.0 # 紹介完了から5秒間は次を検知しない

        self.get_logger().info("Task Planner Node: Host is standing mode - Started.")

    def vision_cb(self, msg):
        """人が来たら、ホストでないことを確認して接近を開始する"""
        if self.state != "WAITING_FOR_GUEST":
            return

        # クールタイムチェック（連続検知防止）
        if (time.time() - self.last_reception_time) < self.cooldown_period:
            return

        data = json.loads(msg.data)
        
        # 誰かが到着したという判定
        if data.get("status") == "guest_arrived" and self.last_vision_status == "searching":
            people = data.get("people", [])
            if not people:
                return

            target_guest = None
            
            # 視界内の人たちをチェック
            for p in people:
                # VisionNodeが計算した map 座標系での位置を取得
                # ※VisionNodeが座標計算していない場合は、dialogue_manager側の計算を待つ
                pos = p.get("map_coords") 
                
                if pos:
                    # ホストの立ち位置からの距離を計算
                    dist_to_host = math.sqrt((pos['x'] - self.HOST_X)**2 + (pos['y'] - self.HOST_Y)**2)
                    
                    # ホストの近く（1m以内）にいる人はスキップ
                    if dist_to_host < self.HOST_IGNORE_RADIUS:
                        self.get_logger().info("Detected person near Host position. Ignoring as Host.")
                        continue
                
                # 最初に見つかった「ホストでない人」をゲストとする
                target_guest = p
                break

            if target_guest:
                self.get_logger().info(f"Real Guest detected! Approaching guest {self.guest_count + 1}...")
                bbox = target_guest.get("bbox", [])

                self.state = "APPROACHING_GUEST"
                
                # DialogueManagerへ接近指示を出す
                instruction = {
                    "action": "APPROACH_GUEST",
                    "data": {"bbox": bbox}
                }
                self.pub_action.publish(String(data=json.dumps(instruction)))
        
        self.last_vision_status = data.get("status")

    def profile_cb(self, msg):
        """NLPで名前・飲み物が確定したら、ホストへの移動指示を出す"""
        profile = json.loads(msg.data)
        self.guest_count += 1
        self.state = "GOING_TO_HOST"

        instruction = {
            "action": "MOVE_TO_HOST",
            "data": {
                "name": profile.get("name"),
                "drink": profile.get("drink"),
                "guest_num": self.guest_count
            }
        }
        self.pub_action.publish(String(data=json.dumps(instruction)))

    def action_status_cb(self, msg):
        """DialogueManagerからの完了報告を受けて状態を遷移させる"""
        status = msg.data

        # 1. ゲストへの接近が完了
        if status == "ARRIVED_AT_GUEST":
            self.state = "RECEPTION"
            # NLP（音声対話）を開始させる
            trigger_msg = String()
            trigger_msg.data = "START_GUEST_RECEPTION"
            self.pub_nlp_trigger.publish(trigger_msg)

        # 2. ホストへの案内と紹介（および指差し）がすべて完了
        elif status == "COMPLETED_GUEST_MANAGEMENT":
            self.last_reception_time = time.time() # クールタイムの計測開始
            
            if self.guest_count < 2:
                # 1人目の後はドアに戻る
                self.get_logger().info("Guest 1 managed. Returning to door for Guest 2...")
                self.state = "RETURNING_TO_DOOR"
                self.pub_action.publish(String(data=json.dumps({"action": "MOVE_TO_DOOR"})))
            else:
                # 2人目完了
                self.get_logger().info("All guests managed. Task Finished!")
                self.state = "FINISHED"

        # 3. ドア（スタート位置）に戻った
        elif status == "ARRIVED_AT_DOOR":
            self.state = "WAITING_FOR_GUEST"
            self.get_logger().info("Waiting for the next guest at the entrance.")

def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()