"""
CV Handover Detection Node
==========================
Monitors a Region of Interest (ROI) near the robot's gripper to detect
when a paper bag is being grasped or released by the human operator.
"""

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class HandoverDetectionNode(Node):
    def __init__(self):
        super().__init__('handover_detection_node')
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        
        self.handover_pub = self.create_publisher(
            Bool,
            '/perception/handover_detected',
            10)
            
        self.bridge = CvBridge()
        
        # Handover ROI (normalized coordinates e.g. center area)
        self.roi_rect = [0.35, 0.4, 0.65, 0.8] # [x1, y1, x2, y2]
        
        # State for motion detection
        self.prev_roi_gray = None
        self.motion_threshold = 5000 # Sensitivity
        
        self.get_logger().info("Handover Detection Node Started")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        
        # Extract ROI
        x1, y1, x2, y2 = [int(self.roi_rect[0]*w), int(self.roi_rect[1]*h), 
                          int(self.roi_rect[2]*w), int(self.roi_rect[3]*h)]
        roi = frame[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (21, 21), 0)
        
        if self.prev_roi_gray is None:
            self.prev_roi_gray = roi_gray
            return
            
        # Detect motion in ROI (simple frame difference)
        frame_delta = cv2.absdiff(self.prev_roi_gray, roi_gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_score = np.sum(thresh)
        
        self.prev_roi_gray = roi_gray
        
        # If significant motion is detected in the handover zone
        if motion_score > self.motion_threshold:
            handover_msg = Bool()
            handover_msg.data = True
            self.handover_pub.publish(handover_msg)
            # self.get_logger().info("Handover activity detected!")

def main(args=None):
    rclpy.init(args=args)
    node = HandoverDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
