import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np

class FollowMeNode(Node):
    def __init__(self):
        super().__init__('follow_me_node')

        self.bridge = CvBridge()
        self.mode = "IDLE"
        self.latest_target = None

        self.min_left = 10.0
        self.min_center = 10.0
        self.min_right = 10.0
        self.avoid_distance = 0.35

        #publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/commands/velocity', 10)

        #subscribers
        self.state_sub = self.create_subscription(String, '/cml_state', self.state_callback, 10)
        self.target_sub = self.create_subscription(Point, '/target_person', self.target_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)


        #control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Follow Me node ready")

    def state_callback(self, msg):
        self.mode = msg.data

    def target_callback(self, msg):
        self.latest_target = msg

    def scan_callback(self, msg):
        ranges = np.array([r if 0.25 < r < 10.0 else 10.0 for r in msg.ranges])
        n = len(ranges)

        if n == 0:
            return

        deg_45 = int(n * (45.0 / 360.0))
        deg_15 = int(n * (15.0 / 360.0))

        self.min_left = np.min(ranges[deg_15:deg_45])
        self.min_center = np.min(np.concatenate((ranges[:deg_15], ranges[-deg_15:])))
        self.min_right = np.min(ranges[-deg_45:-deg_15])

    def control_loop(self):
        # Only move if in FOLLOW mode and we have a valid target
        if self.mode != "FOLLOW" or self.latest_target is None:
            return
        
        twist = Twist()

        # Ensure z-axis is positive for YOLO target lock
        if self.latest_target.z > 0.1:
            # Proportional control for angular velocity based on horizontal error
            center_x = 320  # Assuming 640x480 image
            error_x = self.latest_target.x - center_x

            normalized_error_x = error_x / center_x  # Normalize to [-1, 1]

            twist.angular.z = -normalized_error_x * 0.6

            alignment_factor = max(0.0, 1.0 - abs(normalized_error_x) * 2.0)  # Reduce speed when not aligned

            # Proportional control for linear velocity based on distance error
            distance_error = self.latest_target.z - 0.7
            twist.linear.x = distance_error * 0.4 * alignment_factor

            twist.linear.x = max(min(twist.linear.x, 0.5), -0.2)  # Limit forward speed and allow slight reverse
            twist.angular.z = max(min(twist.angular.z, 0.8), -0.8)  # Limit turning speed

            # Obstacle avoidance logic
            if self.min_center < self.avoid_distance:
                twist.linear.x = min(twist.linear.x, 0.0)  # Stop or reverse if obstacle ahead
            elif self.min_left < self.avoid_distance:
                twist.linear.x *= 0.5  # Slow down when obstacle on the left
                twist.angular.z -= 0.6  # Turn right if obstacle on the left
            elif self.min_right < self.avoid_distance:
                twist.linear.x *= 0.5  # Slow down when obstacle on the right
                twist.angular.z += 0.6  # Turn left if obstacle on the right

            self.cmd_vel_pub.publish(twist)

    def color_callback(self, msg):
        if self.mode != "FOLLOW":
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = cv_image.shape
        center_x = w // 2

        twist = Twist()

        if self.latest_target and self.latest_target.z > 0.1:
            cX = int(self.latest_target.x)
            cY = int(self.latest_target.y)
            cv2.circle(cv_image, (cX, cY), 10, (0, 255, 0), -1)
            cv2.putText(cv_image, f"Distance: {self.latest_target.z:.2f}m", (cX - 50, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Follow", cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FollowMeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()