#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2
import json

from ultralytics import YOLO


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            '/bag_detections',
            10
        )

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame)

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls not in [24, 26, 28]:# backpack, handbag, suitcase
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

                # debug draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # publish detections
        msg_out = String()
        msg_out.data = json.dumps(detections)
        self.publisher.publish(msg_out)

        cv2.imshow("detector", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()