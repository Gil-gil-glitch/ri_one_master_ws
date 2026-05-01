import numpy as np
import pyaudio
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from faster_whisper import WhisperModel


class ASRNode(Node):
    def __init__(self):
        super().__init__('asr_node')

        # ROS2 Publisher
        self.publisher_ = self.create_publisher(String, '/speech_text', 10)

        # Parameters
        self.declare_parameter('whisper_model', 'small')
        self.declare_parameter('language', 'en')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_size', 1024)
        # 環境音に合わせて調整（値が小さいほど感度が高い）
        self.declare_parameter('silence_threshold', 600.0)
        # 無音が何秒続いたら発話終了とするか
        self.declare_parameter('silence_limit', 1.2)
        self.declare_parameter('use_gpu', True)

        self.whisper_model_name = self.get_parameter('whisper_model').value
        self.language = self.get_parameter('language').value
        self.fs = int(self.get_parameter('sample_rate').value)
        self.chunk = int(self.get_parameter('chunk_size').value)
        self.silence_threshold = float(self.get_parameter('silence_threshold').value)
        self.silence_limit = float(self.get_parameter('silence_limit').value)
        self.use_gpu = bool(self.get_parameter('use_gpu').value)

        self.frame_duration = self.chunk / self.fs

        # Audio state
        self.buffer = []
        self.recording = False
        self.silent_time = 0.0

        # Load Whisper
        device = 'cuda' if self.use_gpu else 'cpu'
        compute_type = 'float16' if self.use_gpu else 'int8'
        self.get_logger().info(
            f'Loading Whisper: model={self.whisper_model_name}, device={device}'
        )
        self.stt = WhisperModel(
            self.whisper_model_name, device=device, compute_type=compute_type
        )

        # Audio Input Setup
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.fs,
            input=True,
            frames_per_buffer=self.chunk,
        )

        self.timer = self.create_timer(self.frame_duration, self.audio_callback)
        self.get_logger().info('ASR Node Ready. Listening...')

    # ----------------------------------------------------------
    # ★ 修正: int16 のまま RMS を計算し、float32 の二重変換バグを解消
    # ----------------------------------------------------------
    def is_silent(self, raw_bytes: bytes) -> bool:
        """
        int16 PCM バイト列を受け取り、RMS が閾値未満なら True を返す。
        float32 に変換して再スケールする二重変換を廃止。
        """
        arr = np.frombuffer(raw_bytes, dtype=np.int16)
        rms = np.sqrt(np.mean(arr.astype(np.float32) ** 2))
        # デバッグが必要な時はコメントアウトを外す
        # self.get_logger().info(f"RMS: {rms:.1f}")
        return rms < self.silence_threshold

    def audio_callback(self):
        try:
            data = self.stream.read(self.chunk, exception_on_overflow=False)

            if not self.is_silent(data):
                if not self.recording:
                    self.get_logger().info('Voice detected, recording...')
                self.recording = True
                self.silent_time = 0.0
                # バッファには float32 で積む（Whisper の入力形式）
                frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.buffer.append(frame)
            else:
                if self.recording:
                    self.silent_time += self.frame_duration
                    # 無音部分も少し含める（自然な発話末尾を拾うため）
                    frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.buffer.append(frame)

                    if self.silent_time > self.silence_limit:
                        self.process_utterance()
                        self.buffer = []
                        self.recording = False
                        self.silent_time = 0.0

        except Exception as e:
            self.get_logger().error(f'Audio callback error: {e}')

    def process_utterance(self):
        if not self.buffer:
            return

        try:
            audio = np.concatenate(self.buffer)
            segments, _ = self.stt.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            text = ' '.join(seg.text for seg in segments).strip()

            if text:
                # 誤認識を減らすための軽微なクリーニング
                text = text.replace('.', '').replace('?', '').replace(',', '').strip()
                self.publisher_.publish(String(data=text))
                self.get_logger().info(f'Recognized: "{text}"')
            else:
                self.get_logger().info('Empty speech detected.')

        except Exception as e:
            self.get_logger().error(f'Transcription error: {e}')

    def destroy_node(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ASRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()