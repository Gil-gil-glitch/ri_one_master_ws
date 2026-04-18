import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO


class PersonTargettingNode(Node):

    def __init__(self):
        super().__init__('person_targetting')

        self.bridge = CvBridge()

        self.get_logger().info("Loading YOLO model...")
        self.model = YOLO('yolov8n-seg.pt')
        self.get_logger().info("Model loaded.")

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        self.gesture_sub = self.create_subscription(
            String,
            '/gesture',
            self.gesture_callback,
            10
        )

        # Publisher
        self.target_pub = self.create_publisher(
            Point,
            '/target_person',
            10
        )

        # State
        self.capture_target = False
        self.target_locked = False

        self.prev_center = None
        self.prev_depth = None
        self.latest_depth = None
        
        # Temporal Buffer State (waiting time)
        self.last_seen_time = None

        # Tuning parameters
        self.max_jump_pixels = 120
        self.max_depth_diff = 0.5  # meters
        self.target_timeout = 3.0  # seconds to wait before dropping target completely

    # CALLBACKS
    def gesture_callback(self, msg):
        if msg.data == "pointing":
            if not self.target_locked:
                self.get_logger().info("pointing detected → capturing target")
                self.capture_target = True
            else:
                self.get_logger().info("Gesture detected, but target already locked. Ignoring.")

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')

    # HELPER METHODS
    def get_depth(self, x, y):
        if self.latest_depth is None:
            return None

        h, w = self.latest_depth.shape
        
        # Define a 5x5 window around the center 
        window_size = 2
        x_start = max(0, int(x) - window_size)
        x_end = min(w, int(x) + window_size + 1)
        y_start = max(0, int(y) - window_size)
        y_end = min(h, int(y) + window_size + 1)
        
        region = self.latest_depth[y_start:y_end, x_start:x_end]
        
        # Filter out 0 values (invalid depth)
        valid_depths = region[region > 0]
        
        if len(valid_depths) > 0:
            return np.median(valid_depths) / 1000.0  # mm → meters

        return None


    # Main
    def image_callback(self, msg):
        if self.latest_depth is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if (len(results) > 0 and results[0].boxes is not None) else []

        # CAPTURE TARGET
        if self.capture_target and not self.target_locked:

            best_person = None
            best_score = float('inf')
            center_screen = frame.shape[1] // 2

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                depth = self.get_depth(cx, cy)
                if depth is None:
                    continue

                # prioritize center + closer distance
                score = abs(cx - center_screen) + depth * 100

                if score < best_score:
                    best_score = score
                    best_person = (cx, cy, depth)

            if best_person is not None:
                self.prev_center = (best_person[0], best_person[1])
                self.prev_depth = best_person[2]
                
                # Initialize the timer upon lock
                self.last_seen_time = self.get_clock().now()

                self.target_locked = True
                self.capture_target = False

                self.get_logger().info("TARGET LOCKED")

        # TRACK TARGET
        if self.target_locked:
            best_match = None
            best_score = float('inf')

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                depth = self.get_depth(cx, cy)
                
                if depth is None: continue
                
                pixel_dist = np.sqrt((cx - self.prev_center[0])**2 + (cy - self.prev_center[1])**2)
                depth_diff = abs(depth - self.prev_depth)

                # Relaxed constraints slightly for better real-world tracking
                if pixel_dist < 150 and depth_diff < 0.7:
                    score = pixel_dist + (depth_diff * 200)
                    if score < best_score:
                        best_score = score
                        best_match = (cx, cy, depth)

            if best_match is not None:
                cx, cy, depth = best_match
                self.prev_center, self.prev_depth = (cx, cy), depth
                self.last_seen_time = self.get_clock().now()

                msg_out = Point(x=float(cx), y=float(cy), z=float(depth))
                self.target_pub.publish(msg_out)
                
                # Visual Feedback
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)

            else:
                # Target not found in this frame. Check the buffer timer.
                elapsed_time = (self.get_clock().now() - self.last_seen_time).nanoseconds / 1e9

                if elapsed_time < self.target_timeout:
                    # Still within buffer. Pause movement but keep lock.
                    self.get_logger().warn(f"Target occluded. Waiting... ({elapsed_time:.1f}s / {self.target_timeout}s)")
                    
                    msg_out = Point()
                    msg_out.x = 0.0
                    msg_out.y = 0.0
                    msg_out.z = 0.0 # follow_node ignores z < 0.1, causing robot to stop
                    self.target_pub.publish(msg_out)
                    
                    cv2.putText(frame, f"WAITING... {elapsed_time:.1f}s", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                else:
                    # Buffer expired. Drop the lock completely.
                    self.get_logger().error("Target lost for too long. Dropping lock.")
                    self.target_locked = False
                    self.capture_target = False
                    
                    msg_out = Point()
                    msg_out.x = 0.0
                    msg_out.y = 0.0
                    msg_out.z = -1.0 # Force follow_node to halt
                    self.target_pub.publish(msg_out)

        cv2.imshow("Person Targetting", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = PersonTargettingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()