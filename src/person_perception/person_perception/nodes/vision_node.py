"""
Vision Node: Person Detection ROS 2 Node
==========================================
Dedicated ROS 2 node for person detection using YOLO.
Publishes detected persons (bounding boxes + attributes) to /person_detection.

Part of the 3-node Person Learning System architecture:
  /vision_node -> /person_tracker -> /task_planner
"""

import json
import time
import base64
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

# Conditional import for ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object
    String = object  # Placeholder for type annotations

from ..core.vision import VisionProcessor


class VisionNode(Node):
    """
    ROS 2 Node for person detection.
    
    Responsibility:
    - Detect persons from camera feed (RealSense or webcam)
    - Output bounding boxes with confidence and attributes
    - Publish detections to /person_detection topic
    
    This node does NOT do identity recognition - that's /person_tracker's job.
    """
    
    def __init__(self):
        if ROS2_AVAILABLE:
            super().__init__('vision_node')
            
            # Declare parameters
            self.declare_parameter('model_path', 'yolov8n.pt')
            self.declare_parameter('conf_threshold', 0.25)
            self.declare_parameter('publish_rate', 30.0)
            self.declare_parameter('show_debug_window', True)
            self.declare_parameter('use_webcam', False)
            self.declare_parameter('camera_id', 0)
            
            # Get parameters
            model_path = self.get_parameter('model_path').value
            conf_threshold = self.get_parameter('conf_threshold').value
            publish_rate = self.get_parameter('publish_rate').value
            self.show_debug = self.get_parameter('show_debug_window').value
            self.use_webcam = self.get_parameter('use_webcam').value
            self.camera_id = self.get_parameter('camera_id').value
        else:
            # Standalone mode defaults
            model_path = 'yolov8n.pt'
            conf_threshold = 0.25
            publish_rate = 30.0
            self.show_debug = True
            self.use_webcam = True
            self.camera_id = 0
        
        # Initialize vision processor
        self._log('Initializing VisionProcessor (YOLO)...')
        
        if self.use_webcam:
            self.vision = None  # Will use webcam-based processor
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self._log('ERROR: Failed to open webcam', level='error')
                raise RuntimeError(f'Failed to open webcam {self.camera_id}')
            self._log(f'Webcam {self.camera_id} opened successfully')
            
            # Initialize YOLO model directly for webcam mode
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
        else:
            self.vision = VisionProcessor(
                model_path=model_path,
                conf_threshold=conf_threshold
            )
            self.cap = None
            self.model = None
        
        # Publisher (ROS 2 only)
        self.publisher = None
        if ROS2_AVAILABLE:
            self.publisher = self.create_publisher(
                String,
                '/person_detection',
                10
            )
        
        # Debug window
        if self.show_debug:
            self.window_name = "Vision Node - Person Detection"
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 960, 720)
            except Exception as e:
                self._log(f'OpenCV window init failed: {e}', level='warn')
                self.show_debug = False
        
        # Timer (ROS 2 only)
        if ROS2_AVAILABLE:
            self.timer = self.create_timer(
                1.0 / publish_rate, self.process_callback
            )
        
        self._log('Vision Node started!')
    
    def _log(self, msg: str, level: str = 'info'):
        """Log message via ROS 2 or print."""
        if ROS2_AVAILABLE and hasattr(self, 'get_logger'):
            logger = self.get_logger()
            if level == 'error':
                logger.error(msg)
            elif level == 'warn':
                logger.warn(msg)
            else:
                logger.info(msg)
        else:
            print(f'[VisionNode] [{level.upper()}] {msg}')
    
    def _get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get a frame from camera. Returns (color_image, depth_image)."""
        if self.use_webcam:
            ret, frame = self.cap.read()
            if not ret:
                return None, None
            return frame, None
        else:
            return self.vision.get_frames()
    
    def _detect_persons(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Detect persons in the image.
        
        Returns:
            List of person detections, each containing:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - distance_m: float or None
            - attributes: list of attribute strings
        """
        if self.use_webcam:
            return self._detect_persons_webcam(color_image)
        else:
            return self._detect_persons_realsense(color_image, depth_image)
    
    def _detect_persons_webcam(self, image: np.ndarray) -> List[Dict]:
        """Detect persons using webcam (no depth)."""
        PERSON_CLASS_ID = 0
        results = self.model(image, verbose=False, conf=self.conf_threshold)
        
        persons = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == PERSON_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    # Detect attributes (shirt color, accessories)
                    attributes = self._detect_attributes_simple(
                        image, (x1, y1, x2, y2)
                    )
                    
                    persons.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': round(conf, 4),
                        'distance_m': None,
                        'attributes': attributes
                    })
        
        # Sort by area (largest first = closest)
        persons.sort(
            key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]),
            reverse=True
        )
        return persons
    
    def _detect_persons_realsense(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray]
    ) -> List[Dict]:
        """Detect persons using RealSense (with depth)."""
        raw_persons = self.vision.detect_persons(color_image, depth_image)
        
        persons = []
        for p in raw_persons:
            attributes = []
            try:
                attributes = self.vision.detect_attributes(
                    color_image, p['bbox']
                )
            except Exception:
                pass
            
            persons.append({
                'bbox': list(p['bbox']),
                'confidence': round(p['confidence'], 4),
                'distance_m': round(p.get('distance_m', 0), 3) if p.get('distance_m') else None,
                'attributes': attributes
            })
        
        return persons
    
    def _detect_attributes_simple(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> List[str]:
        """Simple attribute detection (shirt color) without CLIP."""
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        
        # Extract torso region (middle 40% of person height)
        torso_y1 = y1 + int(h * 0.25)
        torso_y2 = y1 + int(h * 0.65)
        torso_x1 = x1 + int(w * 0.2)
        torso_x2 = x2 - int(w * 0.2)
        
        if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
            return []
        
        torso = image[torso_y1:torso_y2, torso_x1:torso_x2]
        if torso.size == 0:
            return []
        
        # Classify shirt color using HSV
        color = self._classify_shirt_color(torso)
        return [f'{color}_Shirt'] if color else []
    
    def _classify_shirt_color(self, crop: np.ndarray) -> Optional[str]:
        """Classify shirt color using HSV analysis."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        mean_h = np.mean(h)
        
        # Low saturation = black/white/grey
        if mean_s < 40:
            if mean_v < 80:
                return 'Black'
            elif mean_v > 180:
                return 'White'
            else:
                return 'Grey'
        
        # Color classification by hue
        if mean_h < 10 or mean_h > 170:
            return 'Red'
        elif 10 <= mean_h < 25:
            return 'Orange'
        elif 25 <= mean_h < 35:
            return 'Yellow'
        elif 35 <= mean_h < 85:
            return 'Green'
        elif 85 <= mean_h < 130:
            return 'Blue'
        elif 130 <= mean_h < 170:
            return 'Purple'
        
        return None
    
    def process_callback(self):
        """Main processing callback — runs at configured rate."""
        color_image, depth_image = self._get_frame()
        if color_image is None:
            return
        
        # Detect persons
        persons = self._detect_persons(color_image, depth_image)
        
        # Encode image to base64 if needed for identity recognition
        image_base64 = ""
        try:
            _, buffer = cv2.imencode('.jpg', color_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self._log(f'Image encoding failed: {e}', level='warn')
        
        # Build message
        msg_data = {
            'timestamp': int(time.time() * 1000),
            'persons': persons,
            'image_base64': image_base64
        }
        
        # Publish (ROS 2)
        if self.publisher is not None:
            msg = String()
            msg.data = json.dumps(msg_data)
            self.publisher.publish(msg)
        
        # Debug visualization
        if self.show_debug:
            self._visualize(color_image, persons)
        
        return msg_data
    
    def _visualize(self, image: np.ndarray, persons: List[Dict]):
        """Draw detection boxes and info."""
        annotated = image.copy()
        
        cv2.putText(
            annotated, f"Persons Detected: {len(persons)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person['bbox']
            conf = person['confidence']
            
            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label = f"Person {i} ({conf:.2f})"
            if person.get('distance_m'):
                label += f" {person['distance_m']:.1f}m"
            
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            
            # Attributes
            attrs = person.get('attributes', [])
            if attrs:
                attr_text = ', '.join(attrs)
                cv2.putText(
                    annotated, attr_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
                )
        
        cv2.imshow(self.window_name, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            self._log('ESC pressed. Shutting down...')
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown."""
        if self.cap is not None:
            self.cap.release()
        
        if self.vision is not None and hasattr(self.vision, 'pipeline'):
            try:
                self.vision.pipeline.stop()
            except Exception:
                pass
        
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        if ROS2_AVAILABLE:
            try:
                self.timer.cancel()
            except Exception:
                pass
            rclpy.shutdown()


def main(args=None):
    """Entry point for the vision node."""
    if not ROS2_AVAILABLE:
        print("=" * 60)
        print("WARNING: ROS 2 not detected — running in standalone mode")
        print("=" * 60)
        
        node = VisionNode()
        try:
            while True:
                result = node.process_callback()
                if result is None:
                    break
        except KeyboardInterrupt:
            print("\nShutting down Vision Node...")
        finally:
            node.shutdown()
        return
    
    rclpy.init(args=args)
    node = VisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
