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

        if self.model.task != "segment":
            self.get_logger().error("Loaded model is not configured for segmentation. Please check the model path and configuration.")
            raise ValueError("Model must be configured for segmentation with task='segment'")
        
        else:
            self.get_logger().info("Model task confiemed: " + self.model.task)

        self.timer = self.create_timer(2.0, self.debug_state)

        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # triggered by voice commands

        self.command_sub = self.create_subscription(
            String, '/voice_imperatives', self.command_callback, 10)
        
        self.target_pub = self.create_publisher(Point, '/target_person', 10)

        # State
        self.capture_target = False
        self.target_locked = False
        self.prev_center = None
        self.prev_depth = None
        self.latest_depth = None

        self.capture_candidates = []
        self.capture_confirm_frames = 5

        self.last_seen_time = None
        self.target_timeout = 3.0

    def debug_state(self):
        self.get_logger().info(
            f"capture_target={self.capture_target} | "
            f"target_locked={self.target_locked} | "
            f"depth_ready={self.latest_depth is not None} | "
            f"candidates={len(self.capture_candidates)}"
        )

    def command_callback(self, msg):
        if msg.data == "following":
            if not self.target_locked:
                self.get_logger().info("Voice command 'follow' detected → capturing target")
                self.capture_target = True
                self.capture_candidates = []
            else:
                self.get_logger().info("Already locked. Ignoring.")
        elif msg.data == "returning":
            if self.target_locked:
                self.get_logger().info("Releasing target lock.")
                self.capture_target = False
                self.target_locked = False
                self.prev_center = None
                self.prev_depth = None
                self.capture_candidates = []
            else:
                self.get_logger().info("Returning command but no lock. Ignoring.")

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_depth(self, x, y):
        if self.latest_depth is None:
            return None

        h, w = self.latest_depth.shape
        window_size = 2
        x_start = max(0, int(x) - window_size)
        x_end = min(w, int(x) + window_size + 1)
        y_start = max(0, int(y) - window_size)
        y_end = min(h, int(y) + window_size + 1)

        region = self.latest_depth[y_start:y_end, x_start:x_end]
        valid_depths = region[region > 0]

        if len(valid_depths) > 0:
            return np.median(valid_depths) / 1000.0
        return None

    def get_mask_centroid(self, mask, frame_shape):
        """
        Compute the true centroid of the segmentation mask.
        Much more stable than the bounding box center, especially
        for partially visible or seated people.
        """
        if mask is None:
            return None, None

        # Resize mask to frame dimensions and threshold to binary
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (frame_shape[1], frame_shape[0])
        )
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Use image moments to find the true center of mass
        M = cv2.moments(mask_binary)
        if M["m00"] == 0:
            return None, None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    def get_detections(self, results, frame):
        """
        Extract (cx, cy, depth, mask) for every detected person,
        preferring mask centroid over box centroid.
        """
        detections = []

        has_masks = (
            results[0].masks is not None and
            len(results[0].masks.data) > 0
        )

        if has_masks:
            for i, mask in enumerate(results[0].masks.data.cpu().numpy()):
                cx, cy = self.get_mask_centroid(mask, frame.shape)
                if cx is None:
                    continue
                depth = self.get_depth(cx, cy)
                if depth is None:
                    continue
                detections.append((cx, cy, depth, mask))
        else:
            # Fallback to box centroid if masks unavailable
            self.get_logger().warn("No masks available, falling back to box centroid.")
            if results[0].boxes is not None:
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    depth = self.get_depth(cx, cy)
                    if depth is None:
                        continue
                    detections.append((cx, cy, depth, None))

        return detections

    def image_callback(self, msg):
        if self.latest_depth is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame, classes=[0], task = "segment", verbose=False)

        if not results or len(results[0].boxes) == 0:
            detections = []
        else:
            detections = self.get_detections(results, frame)

        center_screen = frame.shape[1] // 2

        # CAPTURE TARGET
        if self.capture_target and not self.target_locked:

            best_person = None
            best_score = float('inf')

            for cx, cy, depth, mask in detections:
                # Prioritize centered person; depth is minor tiebreaker
                score = abs(cx - center_screen) * 2.0 + depth * 10
                if score < best_score:
                    best_score = score
                    best_person = (cx, cy, depth)

            if best_person is not None:
                self.capture_candidates.append(best_person)
                self.get_logger().info(
                    f"Candidate {len(self.capture_candidates)}/{self.capture_confirm_frames} "
                    f"cx={best_person[0]}, depth={best_person[2]:.2f}m"
                )

                if len(self.capture_candidates) >= self.capture_confirm_frames:
                    avg_cx = int(np.mean([c[0] for c in self.capture_candidates]))
                    avg_cy = int(np.mean([c[1] for c in self.capture_candidates]))
                    avg_depth = float(np.mean([c[2] for c in self.capture_candidates]))

                    self.prev_center = (avg_cx, avg_cy)
                    self.prev_depth = avg_depth
                    self.last_seen_time = self.get_clock().now()
                    self.target_locked = True
                    self.capture_target = False
                    self.capture_candidates = []

                    self.get_logger().info(
                        f"TARGET LOCKED — cx={avg_cx}, depth={avg_depth:.2f}m"
                    )
            else:
                self.get_logger().warn("No person detected during capture. Retrying...")

        # TRACK TARGET
        if self.target_locked:
            best_match = None
            best_score = float('inf')

            for cx, cy, depth, mask in detections:
                pixel_dist = np.sqrt(
                    (cx - self.prev_center[0])**2 +
                    (cy - self.prev_center[1])**2
                )
                depth_diff = abs(depth - self.prev_depth)

                if pixel_dist < 150 and depth_diff < 0.7:
                    score = pixel_dist + (depth_diff * 200)
                    if score < best_score:
                        best_score = score
                        best_match = (cx, cy, depth, mask)

            if best_match is not None:
                cx, cy, depth, mask = best_match
                self.prev_center = (cx, cy)
                self.prev_depth = depth
                self.last_seen_time = self.get_clock().now()

                msg_out = Point(x=float(cx), y=float(cy), z=float(depth))
                self.target_pub.publish(msg_out)

                # Draw mask overlay and centroid
                if mask is not None:
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0])
                    )
                    overlay = frame.copy()
                    overlay[mask_resized > 0.5] = [0, 255, 0]
                    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)
                cv2.putText(frame, f"{depth:.2f}m", (cx + 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw center line for alignment reference
                cv2.line(frame, (center_screen, 0),
                         (center_screen, frame.shape[0]), (255, 0, 0), 1)

            else:
                elapsed_time = (
                    self.get_clock().now() - self.last_seen_time
                ).nanoseconds / 1e9

                if elapsed_time < self.target_timeout:
                    self.get_logger().warn(
                        f"Target occluded. Waiting... ({elapsed_time:.1f}s / {self.target_timeout}s)"
                    )
                    self.target_pub.publish(Point(x=0.0, y=0.0, z=0.0))
                    cv2.putText(frame, f"WAITING... {elapsed_time:.1f}s", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                else:
                    self.get_logger().error("Target lost. Dropping lock.")
                    self.target_locked = False
                    self.capture_target = False
                    self.target_pub.publish(Point(x=0.0, y=0.0, z=-1.0))

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