import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GestureIdentificationNode(Node):

    def __init__(self):
        super().__init__('gesture_identification')

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)

        self.publisher_ = self.create_publisher(
            String,
            '/gesture',
            10)

        self.pointing_pub = self.create_publisher(
            Float32MultiArray,
            '/perception/pointing_vector',
            10)

        self.bridge = CvBridge()

        # Load Hand Landmarker
        hand_model_path = "/home/ri-one/ri_one_master_ws/hand_landmarker.task"
        base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options_hand)

        # Load Pose Landmarker for 3D pointing ray-cast
        pose_model_path = "/home/ri-one/ri_one_master_ws/pose_landmarker_heavy.task"
        base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(options_pose)

        self.get_logger().info("Multi-Modal Gesture & Pointing Node Started")

    def is_pointing_gesture(self, hand_landmarks):
        """Index extended, others folded."""
        # Simplified check for index extension vs others
        tips = [12, 16, 20] # Middle, Ring, Pinky
        pips = [10, 14, 18]
        
        index_up = hand_landmarks[8].y < hand_landmarks[6].y
        others_down = all(hand_landmarks[t].y > hand_landmarks[p].y for t, p in zip(tips, pips))
        
        return index_up and others_down

    def calculate_3d_pointing_vector(self, pose_landmarks, hand_landmarks):
        """
        Calculates a 3D ray from the shoulder to the index finger.
        Returns a normalized vector [dx, dy, dz].
        """
        # Right shoulder (12) and Right Index (Hand 8)
        # Note: Landmarker coordinate systems differ. We use Relative Normalized.
        # We'll use the shoulder and wrist/index from PoseLandmarker for consistency
        shoulder = pose_landmarks[12]
        index = pose_landmarks[20] # Right index finger in Pose
        
        direction = np.array([
            index.x - shoulder.x,
            index.y - shoulder.y,
            index.z - shoulder.z
        ])
        
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        return direction / norm

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))

        # 1. Detect Hand Gestures
        hand_result = self.hand_detector.detect(mp_image)
        gesture = "none"
        
        if hand_result.hand_landmarks:
            landmarks = hand_result.hand_landmarks[0]
            if self.is_pointing_gesture(landmarks):
                gesture = "pointing"
            
            gesture_msg = String()
            gesture_msg.data = gesture
            self.publisher_.publish(gesture_msg)

        # 2. If pointing, calculate 3D Vector using PoseLandmarker
        if gesture == "pointing":
            pose_result = self.pose_detector.detect(mp_image)
            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                vector = self.calculate_3d_pointing_vector(landmarks, None)
                if vector is not None:
                    vec_msg = Float32MultiArray()
                    vec_msg.data = [float(v) for v in vector]
                    self.pointing_pub.publish(vec_msg)
                    # self.get_logger().info(f"Published pointing vector: {vec_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = GestureIdentificationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


