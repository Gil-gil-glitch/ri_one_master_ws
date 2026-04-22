#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import json

class DistanceNode(Node):
    def __init__(self):
        super().__init__('distance_node')
        self.bridge = CvBridge()
        
        self.color_image = None
        self.depth_image = None

        # Subscriptions
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(String, '/bag_detections', self.detection_callback, 10)

        # Publisher for the pickup node
        self.publisher = self.create_publisher(String, '/bag_positions_3d', 10)

        self.get_logger().info("Distance Node initialized using RealSense Depth.")

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        # Depth images are usually 16-bit unsigned integers (millimeters)
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def detection_callback(self, msg):
        if self.depth_image is None or self.color_image is None:
            return

        detections = json.loads(msg.data)
        h, w = self.depth_image.shape
        cx_cam, cy_cam = w / 2, h / 2
        
        # RealSense D435 approx intrinsics (Standard for 640x480)
        # If using 1280x720, these should be ~900-1000
        fx, fy = 600.0, 600.0 

        bag_positions_3d = []

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            
            # 1. Get the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # 2. Extract depth from a small 5x5 patch around the center to avoid noise/outliers
            patch = self.depth_image[max(0, center_y-2):min(h, center_y+2), 
                                     max(0, center_x-2):min(w, center_x+2)]
            
            # Filter out 0 (invalid depth) and find the median
            valid_depths = patch[patch > 0]
            if len(valid_depths) == 0:
                continue
                
            distance_mm = np.median(valid_depths)
            z = distance_mm / 1000.0  # Convert mm to meters

            # 3. Project 2D pixel to 3D space using Camera Intrinsics
            x = (center_x - cx_cam) * z / fx
            y = (center_y - cy_cam) * z / fy

            bag_positions_3d.append({"x": float(x), "y": float(y), "z": float(z)})

        # Publish the accurate 3D positions
        msg_out = String()
        msg_out.data = json.dumps(bag_positions_3d)
        self.publisher.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = DistanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()