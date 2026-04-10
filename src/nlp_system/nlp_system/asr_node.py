import numpy as np
import pyaudio

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from faster_whisper import WhisperModel


class ASRNode(Node):
    def __init__(self):
        super().__init__('asr_node')

        # ===== ROS2 Publisher =====
        self.publisher_ = self.create_publisher(String, '/speech_text', 10)

        # ===== Parameters =====
        self.declare_parameter('whisper_model', 'small')
        self.declare_parameter('language', 'en')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_size', 1024)
        self.declare_parameter('silence_threshold', 120.0)
        self.declare_parameter('silence_limit', 1.0)
        self.declare_parameter('use_gpu', True)

        self.whisper_model_name = self.get_parameter('whisper_model').value
        self.language = self.get_parameter('language').value
        self.fs = int(self.get_parameter('sample_rate').value)
        self.chunk = int(self.get_parameter('chunk_size').value)
        self.silence_threshold = float(self.get_parameter('silence_threshold').value)
        self.silence_limit = float(self.get_parameter('silence_limit').value)
        self.use_gpu = bool(self.get_parameter('use_gpu').value)

        self.frame_duration = self.chunk / self.fs

        # ===== Audio state =====
        self.buffer = []
        self.recording = False
        self.silent_time = 0.0

        # ===== Load Whisper =====
        device = 'cuda' if self.use_gpu else 'cpu'
        compute_type = 'float16' if self.use_gpu else 'int8'

        self.get_logger().info(
            f'Loading Whisper model="{self.whisper_model_name}", device="{device}", compute_type="{compute_type}"'
        )

        self.stt = WhisperModel(
            self.whisper_model_name,
            device=device,
            compute_type=compute_type
        )

        # ===== Open microphone =====
        self.audio = pyaudio.PyAudio()

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.fs,
            input=True,
            frames_per_buffer=self.chunk
        )

        # Timer-based polling
        self.timer = self.create_timer(self.frame_duration, self.audio_callback)

        self.get_logger().info('asr_node started.')
        self.get_logger().info('Listening from microphone and publishing text to /speech_text')

    def is_silent(self, arr: np.ndarray) -> bool:
        # 元コードの判定をそのまま利用
        return np.mean(np.abs(arr * 32768)) < self.silence_threshold

    def audio_callback(self):
        try:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if not self.is_silent(frame):
                self.recording = True
                self.silent_time = 0.0
                self.buffer.append(frame)
            else:
                if self.recording:
                    self.silent_time += self.frame_duration

                if self.recording and self.silent_time > self.silence_limit:
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

            segments, info = self.stt.transcribe(
                audio,
                language=self.language,
                vad_filter=True
            )

            text = ' '.join(seg.text for seg in segments).strip()

            if text:
                msg = String()
                msg.data = text
                self.publisher_.publish(msg)

                self.get_logger().info(f'Recognized: "{text}"')
            else:
                self.get_logger().info('No speech recognized.')

        except Exception as e:
            self.get_logger().error(f'Transcription error: {e}')

    def destroy_node(self):
        try:
            if hasattr(self, 'stream') and self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
            if hasattr(self, 'audio') and self.audio is not None:
                self.audio.terminate()
        except Exception as e:
            self.get_logger().warning(f'Audio cleanup warning: {e}')

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ASRNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping asr_node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
