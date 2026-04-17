import rclpy
from rclpy.node import Node
from inference_sdk import InferenceHTTPClient
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import threading
import cv2
from dotenv import load_dotenv
import os
import time

load_dotenv()

# APIクライアント
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv('API_KEY')
)

class PositionPublisher(Node):
    def __init__(self):
        super().__init__('position_publisher')

        # Publisher & Subscriber
        self.publisher_ = self.create_publisher(Float64MultiArray, 'position_topic', 10)
        self.subscriber_ = self.create_subscription(Float64MultiArray, "position_topic", self.position_callback, 10)
        self._color_sub = self.create_subscription(Image, '/image_raw', self._color_callback, 10)

        # OpenCVブリッジと変数
        self._bridge = CvBridge()
        self._color_image = None
        self.lock = threading.Lock()
        self.last_predictions = []

        # 推論スレッド開始
        self.thread = threading.Thread(target=self.inference_thread, daemon=True)
        self.thread.start()

    def position_callback(self, msg):
        self.get_logger().info(f"Subscribed position: {msg.data}")

    def _color_callback(self, msg):
        """ROS画像メッセージをOpenCV形式に変換"""
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self._color_image = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f"Color image conversion failed: {e}")

    def inference_thread(self):
        while rclpy.ok():
            with self.lock:
                if self._color_image is None:
                    continue
                frame = self._color_image.copy()

            # 推論を軽くするため縮小
            resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

            # Roboflow推論
            result = CLIENT.infer(resized_frame, model_id="person-luggage/3")

            with self.lock:
                self.last_predictions = [
                    prediction for prediction in result.get('predictions', [])
                    if prediction['class'] == 'paperbag' and prediction['confidence'] > 0.7
                ]

            if self.last_predictions:
                for prediction in self.last_predictions:
                    position = [float(prediction['x']), float(prediction['y'])]
                    msg = Float64MultiArray()
                    msg.data = position
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'Published position: {position}')

            time.sleep(0.1)  # 少し待ってCPU使用率を抑える

    def run(self):
        """ カメラ映像をリアルタイムで表示し、検出結果を描画 """
        while rclpy.ok():
            with self.lock:
                if self._color_image is None:
                    continue
                image = self._color_image.copy()
                predictions = self.last_predictions.copy()

            # 検出結果を描画
            for pred in predictions:
                x, y, w, h = map(int, [pred['x'], pred['y'], pred['width'], pred['height']])
                x, y, w, h = x * 2, y * 2, w * 2, h * 2  # 縮小分を戻す
                pt1, pt2 = (x - w // 2, y - h // 2), (x + w // 2, y + h // 2)
                cv2.rectangle(image, pt1, pt2, (255, 0, 0), 3)

            cv2.imshow("Camera Window", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = PositionPublisher()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    node.run()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
