import numpy as np
import pyaudio
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from faster_whisper import WhisperModel
import scipy.signal as signal


class ASRNode(Node):
    def __init__(self):
        super().__init__('cml_asr_node')
        # CHANGE: Match the CMLCoordinator's subscriber topic
        self.publisher_ = self.create_publisher(String, '/voice_imperatives', 10)

        # ===== SETTINGS =====
        self.hw_fs = 44100
        self.target_fs = 16000
        self.chunk = 4096
        self.silence_threshold = 700.0
        self.silence_limit = 1.5

        self.stream = None

        # ===== LOAD WHISPER =====
        self.get_logger().info('Loading Whisper model (GPU)...')
        self.stt = WhisperModel(
            'small',
            device='cuda',
            compute_type='float16',
            device_index=0
        )

        self.audio = pyaudio.PyAudio()

        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.hw_fs,
                input=True,
                input_device_index=9,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            self.get_logger().error(f"Hardware Error: {e}")
            return

        self.buffer = []
        self.recording = False
        self.silent_time = 0.0

        self.timer = self.create_timer(self.chunk / self.hw_fs, self.audio_callback)
        self.get_logger().info('ASR Node Ready. Integrated with CML Coordinator.')

    def is_silent(self, arr):
        amplitude = np.mean(np.abs(arr * 32768))
        return amplitude < self.silence_threshold

    def audio_callback(self):
        if self.stream is None: return
        try:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            full_frame = np.frombuffer(data, dtype=np.int16).reshape(-1, 2)
            frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            rms = np.sqrt(np.mean(frame ** 2))
            self.draw_volume_bar(rms)

            if not self.is_silent(frame):
                if not self.recording: print("\n[Voice Detected]")
                self.recording = True
                self.silent_time = 0.0
                self.buffer.append(frame)
            elif self.recording:
                self.silent_time += (self.chunk / self.hw_fs)
                self.buffer.append(frame)
                if self.silent_time > self.silence_limit:
                    self.process_utterance()
                    self.buffer, self.recording, self.silent_time = [], False, 0.0
        except Exception as e:
            self.get_logger().error(f'Audio Error: {e}')

    def draw_volume_bar(self, rms):
        bar_length = 30
        level = int(rms * bar_length * 15.0)
        if level > bar_length: level = bar_length
        bar = "█" * level + "-" * (bar_length - level)
        status = "REC" if self.recording else "IDL"
        print(f"\r[{status}] |{bar}| {rms:.4f}", end="", flush=True)

    def process_utterance(self):
        if not self.buffer: return

        print("\n--- Transcription ---")
        audio_data = np.concatenate(self.buffer)
        num_samples = int(len(audio_data) * self.target_fs / self.hw_fs)
        audio_16k = signal.resample(audio_data, num_samples)

        segments, _ = self.stt.transcribe(
            audio_16k,
            language='en',
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        text = ' '.join(seg.text for seg in segments).strip().lower()

        if text:
            self.get_logger().info(f'Heard: "{text}"')
            msg = String()

            # --- SIMPLE KEYWORD LOGIC ---
            if "follow" in text:
                msg.data = "following"
                self.publisher_.publish(msg)
                self.get_logger().info("Command Sent: following")

            elif "thank you" in text or "thanks" in text:
                msg.data = "returning"
                self.publisher_.publish(msg)
                self.get_logger().info("Command Sent: returning")

    def destroy_node(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        super().destroy_node()


def main():
    rclpy.init()
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