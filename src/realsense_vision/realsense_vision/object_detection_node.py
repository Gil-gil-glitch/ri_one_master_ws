"""
Zero-Shot Object Detection Node (YOLO-World v2)
================================================
ROS 2 node for real-time zero-shot detection of task-relevant objects
(Paper Bags, Chairs) using YOLO-World v2.

Publishes detections to /perception/objects with bounding boxes and
depth-estimated distances from the RealSense camera.

No custom training required — objects are specified via text prompts.
"""

import json
from typing import List, Dict, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


# Default prompts for competition tasks
DEFAULT_OBJECT_PROMPTS = ["paper bag", "chair", "person"]


class ObjectDetectionNode(Node):
    """
    ROS 2 Node for zero-shot object detection using YOLO-World v2.

    Subscribes: /camera/camera/color/image_raw (optional, or uses RealSense directly)
    Publishes:  /perception/objects (JSON with detections)
    """

    def __init__(self):
        super().__init__('object_detection_node')

        # Parameters
        self.declare_parameter('model_path', 'yolov8s-worldv2.pt')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('publish_rate', 15.0)
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter(
            'object_prompts',
            DEFAULT_OBJECT_PROMPTS
        )
        self.declare_parameter('use_realsense', True)

        model_path = self.get_parameter('model_path').value
        conf_threshold = self.get_parameter('conf_threshold').value
        publish_rate = self.get_parameter('publish_rate').value
        self.show_debug = self.get_parameter('show_debug_window').value
        self.object_prompts = self.get_parameter('object_prompts').value
        use_realsense = self.get_parameter('use_realsense').value

        if not YOLO_AVAILABLE:
            self.get_logger().error('ultralytics not installed!')
            return

        # Load YOLO-World model and set custom classes
        self.get_logger().info(
            f'Loading YOLO-World: {model_path}'
        )
        self.model = YOLO(model_path)
        self.model.set_classes(self.object_prompts)
        self.get_logger().info(
            f'Zero-shot prompts: {self.object_prompts}'
        )

        self.conf_threshold = conf_threshold

        # RealSense setup
        self.pipeline = None
        self.align = None
        self.depth_scale = 1.0
        self.bridge = CvBridge()
        self._current_depth = None

        if use_realsense and REALSENSE_AVAILABLE:
            try:
                self.pipeline = rs.pipeline()
                rs_config = rs.config()
                rs_config.enable_stream(
                    rs.stream.color, 640, 480, rs.format.bgr8, 30
                )
                rs_config.enable_stream(
                    rs.stream.depth, 640, 480, rs.format.z16, 30
                )
                profile = self.pipeline.start(rs_config)
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                self.align = rs.align(rs.stream.color)
                self.get_logger().info('RealSense pipeline started.')
            except Exception as e:
                self.get_logger().warn(
                    f'RealSense init failed: {e}. Using topic mode.'
                )
                self.pipeline = None

        # If no RealSense, subscribe to image topic
        if self.pipeline is None:
            self.image_sub = self.create_subscription(
                Image,
                '/camera/camera/color/image_raw',
                self._image_callback,
                10
            )
            self._latest_frame = None

        # Publisher
        self.publisher = self.create_publisher(
            String, '/perception/objects', 10
        )

        # Debug window
        if self.show_debug:
            self.window_name = "YOLO-World Zero-Shot"
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 960, 540)
            except Exception:
                self.show_debug = False

        # Timer
        self.timer = self.create_timer(
            1.0 / publish_rate, self._process_callback
        )
        self.get_logger().info('Object Detection Node started!')

    def _image_callback(self, msg: Image):
        """Receive frames from ROS topic when RealSense is unavailable."""
        self._latest_frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8'
        )

    def _get_frames(self):
        """Get color and depth frames."""
        if self.pipeline is not None:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color:
                return None, None
            color_img = np.asanyarray(color.get_data())
            depth_img = (
                np.asanyarray(depth.get_data()) if depth else None
            )
            return color_img, depth_img
        elif hasattr(self, '_latest_frame') and self._latest_frame is not None:
            return self._latest_frame, None
        return None, None

    def _get_depth_at(
        self, depth_image: np.ndarray, cx: int, cy: int
    ) -> Optional[float]:
        """Get median depth in a small ROI around (cx, cy)."""
        if depth_image is None:
            return None
        h, w = depth_image.shape[:2]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        roi = depth_image[
            max(0, cy - 3): min(h, cy + 4),
            max(0, cx - 3): min(w, cx + 4)
        ]
        if roi.size == 0:
            return None
        return float(np.median(roi)) * self.depth_scale

    def _process_callback(self):
        """Main detection loop."""
        color_image, depth_image = self._get_frames()
        if color_image is None:
            return

        # Run YOLO-World inference
        results = self.model(
            color_image,
            conf=self.conf_threshold,
            verbose=False
        )

        detections: List[Dict] = []
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls_id = int(det.cls[0])
            class_name = self.object_prompts[cls_id] if cls_id < len(
                self.object_prompts
            ) else f"class_{cls_id}"

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            distance = self._get_depth_at(depth_image, cx, cy)

            detections.append({
                'class_name': class_name,
                'confidence': round(conf, 4),
                'bbox': [x1, y1, x2, y2],
                'center': [cx, cy],
                'distance_m': round(distance, 3) if distance else None,
            })

        # Publish
        msg_data = {
            'timestamp': self.get_clock().now().nanoseconds,
            'object_count': len(detections),
            'objects': detections,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.publisher.publish(msg)

        # Debug
        if self.show_debug:
            self._visualize(color_image, detections)

    def _visualize(
        self, image: np.ndarray, detections: List[Dict]
    ):
        """Draw debug overlay."""
        annotated = image.copy()
        colors = {
            'paper bag': (0, 200, 255),
            'chair': (255, 150, 0),
            'person': (0, 255, 0),
        }
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = colors.get(det['class_name'], (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            if det['distance_m']:
                label += f" | {det['distance_m']:.2f}m"
            cv2.putText(
                annotated, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        cv2.putText(
            annotated,
            f"YOLO-World | {len(detections)} objects",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow(self.window_name, annotated)
        cv2.waitKey(1)

    def shutdown(self):
        """Clean shutdown."""
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
