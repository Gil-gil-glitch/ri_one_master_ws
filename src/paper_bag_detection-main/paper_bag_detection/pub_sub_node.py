import rclpy
from rclpy.node import Node
from inference_sdk import InferenceHTTPClient
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import threading
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

# APIクライアント
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv('API_KEY')
)

class PositionPublisher(Node):
    def __init__(self):
        super().__init__('person_bag_position')  # ノード名
        self.publisher_ = self.create_publisher(Float64MultiArray, 'position_topic', 10)  # 送信するトピック
        # self.subscription = self.create_subscription(String, 'switch_topic', self.listener_callback, 10)  # 送信するトピック
        self.subscriber_ = self.create_subscription(Float64MultiArray, "position_topic", self.position_callback, 10) # 送信するトピック
        
        self.lock = threading.Lock()  # スレッド間の競合を防ぐ
        self.last_predictions = []  # 最新の検出結果
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.switch_flag: str = None  # 初期状態

        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
            exit()

        # スレッド開始
        self.thread = threading.Thread(target=self.inference_thread, daemon=True)
        self.thread.start()

    def position_callback(self,msg):
            self.get_logger().info(f"Subscribed position:{msg.data}") 

    # def listener_callback(self, msg):
    #         self.get_logger().info(f'Received: {msg.data}')  # 受信したメッセージを表示
    #         if msg.data == "paperbag":
    #             self.switch_flag = "paperbag"
    #         elif msg.data == "person": 
    #             self.switch_flag = "person"
    #         elif msg.data == "stop":
    #             self.switch_flag = None
    #         self.get_logger().info(f'Switched detection target to: {self.switch_flag}')
            
    def inference_thread(self):
        """ 推論を実行するスレッド """
        while rclpy.ok():  # ROS2が動作中かチェック
            if not self.switch_flag: continue

            success, frame = self.cap.read()
            if (not success) or (frame is None):
                self.get_logger().warn("Failed to read frame from camera.")
                continue
            
            # 画像を縮小して推論を軽量化
            resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            result = CLIENT.infer(resized_frame, model_id="person-luggage/3")

            with self.lock:
                self.last_predictions = [
                    prediction for prediction in result.get('predictions', [])
                    if prediction['class'] == self.switch_flag and prediction['confidence'] > 0.7
                ]
                
                # もし信頼度70%以上のデータがあればパブリッシュ
                if self.last_predictions:
                    for prediction in self.last_predictions:
                        # normalized_x = prediction['x'] / (frame.shape[1] // 2)
                        # normalized_y = prediction['y'] / (frame.shape[0] // 2) 
                        msg = Float64MultiArray()
                        msg.data = [prediction['x'], prediction['y']]
                        self.publisher_.publish(msg)
                        self.get_logger().info(f'Published {self.switch_flag} position: {prediction["x"]}, {prediction["y"]}')
                        
    def run(self):
        """ カメラ映像を表示しながらリアルタイム処理 """
        while rclpy.ok():
            success, image = self.cap.read()
            if not success or image is None:
                self.get_logger().warn("Failed to read frame from camera.")
                continue

            with self.lock:
                for prediction in self.last_predictions:
                    x, y, w, h = map(int, [prediction['x'], prediction['y'], prediction['width'], prediction['height']])
                    x, y, w, h = x * 2, y * 2, w * 2, h * 2  # 画像拡大
                    pt1, pt2 = (x - w // 2, y - h // 2), (x + w // 2, y + h // 2)
                    cv2.rectangle(image, pt1, pt2, (255, 0, 0), 3)  # 青枠描画

            cv2.imshow('Camera Window', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 終了処理
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = PositionPublisher()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,),daemon=True)
    spin_thread.start()

    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
