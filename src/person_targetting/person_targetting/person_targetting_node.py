#
#
#   person_targetting_node
#
#   This node is responsible for determining what person the robot needs to lock on
#   for the Carry My Luggage task
#
#

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import numpy as np
import mediapipe as mp
from ultralytics import YOLO

from mediapipe.tasks import python
from mediapipe.tasks.python import 

class PersonTargettingNode(Node):
    super().__init__("person_targetting")

    self.bridge = CvBridge

    self.get_logger().info("Loading YOLO model ...")
    self.model = YOLO("yolov8n-seg.pt")
    self.get_logger().info("Model loaded")

    #Subscribing

    self.subscription = self.create_subscription(
        Image,
        '/camera/camera/color/image_raw',
        self.image_callback,
        10)

    #Publishing

    self.target_pub = self.create_publisher(
        String,
        "/target_person",
        10
    )

    #Temporary  