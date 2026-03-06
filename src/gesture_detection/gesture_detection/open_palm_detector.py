import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class OpenPalmDetector(Node):

    def __init__(self):
        super().__init__('open_palm_detector')

        self.subscription = self.create_subscription(
            Image,
            '/kinect/rgb/image_raw',
            self.image_callback,
            10)

        self.bridge = CvBridge()

        # ---- MediaPipe Setup ----
        model_path = "/home/ri-one/ros2_ws_test/hand_landmarker.task"

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        self.get_logger().info("Open Palm Detector Node Started")

        self.publisher_ = self.create_publisher(
            String,
            "/gesture",
            10
        )

    def is_open_palm(self, landmarks):
        """
        landmarks: NormalizedLandmark list
        """

        # Index, middle, ring, pinky
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        fingers_extended = 0

        for tip, pip in zip(tips, pips):
            if landmarks[tip].y < landmarks[pip].y:
                fingers_extended += 1

        # Thumb (basic horizontal check)
        if landmarks[4].x > landmarks[3].x:
            fingers_extended += 1

        return fingers_extended == 5


    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.array(frame)
        )

        result = self.detector.detect(mp_image)

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                if self.is_open_palm(hand_landmarks):
                    print("Open palm detected")
                    
                    self.get_logger().info("Open palm detected")
                    msg = String()
                    msg.data = "open_palm"
                    self.publisher_.publish(msg)
                    


def main(args=None):
    rclpy.init(args=args)
    node = OpenPalmDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()