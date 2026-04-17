import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os

class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')
        self.subscription = self.create_subscription(
            String, '/robot_speech', self.listener_callback, 10)
        
        # モデルのパスを指定
        self.model_path = os.path.expanduser("~/models/en_US-joe-medium.onnx")
        self.get_logger().info('High-quality Offline TTS (Piper) started.')

    def listener_callback(self, msg):
        text = msg.data
        if not text:
            return

        self.get_logger().info(f'Speaking (Offline): "{text}"')
        
        try:
            # Piperコマンドを呼び出し、標準出力をaplay（再生コマンド）に渡す
            # shell=Trueでパイプライン処理を実行
            cmd = f'echo "{text}" | piper --model {self.model_path} --output_raw | aplay -r 22050 -f S16_LE -t raw'
            subprocess.run(cmd, shell=True)
            
        except Exception as e:
            self.get_logger().error(f'Piper TTS Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()