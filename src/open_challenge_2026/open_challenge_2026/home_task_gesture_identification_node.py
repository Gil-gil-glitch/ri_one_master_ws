#
##   Home Task Gesture Identification Node
#
#   This node subscribes to the RealSense camera feed and publishes the to the /gesture topic when it detects a 
#   thumbsup. The thumbsup will be used to trigger the follow_state on the home_task_coordinator node, which 
#   will cause the robot to lock onto a particular person and follow them around for 5 seconds.  
#
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HomeTaskGestureIdentificationNode(Node):
    def __init__(self):
        super().__init__('home_task_gesture_identification')
        self.subscription = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(String, '/gesture', 10)
        self.bridge = CvBridge()

        model_path = "/home/ri-one/ri_one_master_ws/hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.get_logger().info("Home Task Gesture Identification Node Started (Directional)")


    def is_thumbsup(self, landmarks):
        """Calculate distance from wrist (0) to tips to see if they are extended. Returns 
        True if thumb is extended and all other fingers are folded."""
    
        def dist(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

        wrist = landmarks[0]
        thumb_is_extended = dist(wrist, landmarks[4]) > dist(wrist, landmarks[2]) * 1.5
        
        # Check if other fingers are folded (tip closer to wrist than middle joint)
        index_folded = dist(wrist, landmarks[8]) < dist(wrist, landmarks[6])
        middle_folded = dist(wrist, landmarks[12]) < dist(wrist, landmarks[10])
        ring_folded = dist(wrist, landmarks[16]) < dist(wrist, landmarks[14])
        pinky_folded = dist(wrist, landmarks[20]) < dist(wrist, landmarks[18])
        
        return thumb_is_extended and index_folded and middle_folded and ring_folded and pinky_folded


    def image_callback(self, msg):
        """Convert ROS image to OpenCV format, run hand landmark detection, and publish gesture if thumbsup is detected."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            # If this logs constantly, the detector isn't finding any hands
            # Check topic: ros2 topic echo /camera/camera/color/image_raw
            return 

        for hand_landmarks in result.hand_landmarks:
            if self.is_thumbsup(hand_landmarks):
                self.get_logger().info("GESTURE DETECTED: thumbsup")
                
                gesture_msg = String()
                gesture_msg.data = "thumbsup"
                self.publisher_.publish(gesture_msg)


def main(args=None):

    rclpy.init(args=args)

    node = HomeTaskGestureIdentificationNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()

