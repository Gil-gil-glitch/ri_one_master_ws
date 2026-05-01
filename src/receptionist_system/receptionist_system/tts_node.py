import queue
import subprocess
import threading
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')

        self.subscription = self.create_subscription(
            String, '/robot_speech', self.listener_callback, 10
        )

        # モデルのパスを指定
        self.model_path = os.path.expanduser("~/models/en_US-joe-medium.onnx")

        # ★ 修正: ブロッキング subprocess を非同期キューに変更
        #   複数の発話が連続しても順番通りに再生され、
        #   次のメッセージを取りこぼさない。
        self._speech_queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker.start()

        self.get_logger().info('High-quality Offline TTS (Piper) started.')

    def listener_callback(self, msg):
        text = msg.data.strip()
        if text:
            self._speech_queue.put(text)

    def _speech_worker(self):
        """バックグラウンドスレッドでキューを順番に処理する"""
        while True:
            try:
                text = self._speech_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self.get_logger().info(f'Speaking: "{text}"')
            try:
                # シェルインジェクション対策: テキストにシングルクォートが含まれる場合をエスケープ
                safe_text = text.replace("'", "'\\''")
                cmd = (
                    f"echo '{safe_text}' | "
                    f"piper --model {self.model_path} --output_raw | "
                    f"aplay -r 22050 -f S16_LE -t raw"
                )
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f'Piper TTS Error: {e}')
            except Exception as e:
                self.get_logger().error(f'TTS unexpected error: {e}')
            finally:
                self._speech_queue.task_done()


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()