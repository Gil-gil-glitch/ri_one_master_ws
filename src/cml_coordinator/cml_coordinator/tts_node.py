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
        
        self.model_path = os.path.expanduser("~/models/en_US-joe-medium.onnx")
        self.get_logger().info('High-quality Offline TTS (Piper) started with Popen.')
        
        # 現在実行中の音声プロセスを保持する変数
        self.process = None

    def listener_callback(self, msg):
        text = msg.data
        if not text:
            return

        # 前の音声がまだ流れていたら、止めるか終わるのを待つ
        if self.process is not None:
            if self.process.poll() is None: # poll()がNoneならまだ実行中
                self.get_logger().warn('Previous speech is still playing. Overlapping...')
                # 必要なら self.process.terminate() で強制終了させることも可能

        self.get_logger().info(f'Speaking (Non-blocking): "{text}"')
        
        try:
            cmd = f'echo "{text}" | piper --model {self.model_path} --output_raw | aplay -r 22050 -f S16_LE -t raw'
            
            # Popenで実行（shell=Trueでパイプラインを有効にする）
            # これにより、この関数は即座に終了し、ROSの処理に戻れる
            self.process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
            
        except Exception as e:
            self.get_logger().error(f'Piper TTS Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    # 終了時にプロセスが残っていたら掃除
    if node.process and node.process.poll() is None:
        node.process.terminate()
    node.destroy_node()
    rclpy.shutdown()
