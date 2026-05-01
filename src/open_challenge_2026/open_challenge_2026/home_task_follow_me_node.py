#
## Home Task Follow Me Node
#
#   This node implements the "Follow Me" behavior for the home task. It subscribes to the 
#   /cml_state topic to know when to enter FOLLOW mode, and to the /target_person topic 
#   for the (x, y, z) coordinates of the person to follow. It also subscribes to the 
#   /scan topic for obstacle avoidance and the camera feed for visual debugging. The 
#   node uses a PID controller to compute velocity commands that keep the robot 
#   following the target person at a comfortable distance while avoiding obstacles. 
#   It includes safety features like timeouts and smooths sensor data to prevent 
#   jittery movements.
#
#
#
#

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

        self.camera_center_x = None  

        # Obstacle avoidance
        self.min_left = 10.0
        self.min_center = 10.0
        self.min_right = 10.0
        self.avoid_distance = 0.35

        # Timeout safety
        self.last_target_time = self.get_clock().now()
        self.target_timeout = 0.5  

        # === PID PARAMETERS ===

        # Smoothing values
        self.smoothed_x = None
        self.smoothed_z = None
        self.alpha = 0.3  # Smoothing factor for low-pass filter
        
        # Angular PID (turning)
        self.kp_ang = 1.2
        self.ki_ang = 0.0
        self.kd_ang = 0.4

        self.integral_ang = 0.0
        self.prev_error_ang = 0.0

        # Linear PID (distance)
        self.kp_lin = 0.6
        self.ki_lin = 0.0
        self.kd_lin = 0.2

        self.integral_lin = 0.0
        self.prev_error_lin = 0.0

        self.prev_time = self.get_clock().now()

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/commands/velocity', 10)

        # Subscribers
        self.state_sub = self.create_subscription(String, '/cml_state', self.state_callback, 10)
        self.target_sub = self.create_subscription(Point, '/target_person', self.target_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)

        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Follow Me node with PID ready")

    def state_callback(self, msg):
        self.mode = msg.data

    def target_callback(self, msg):
        self.latest_target = msg
        self.last_target_time = self.get_clock().now()

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
        twist = Twist()

        if self.camera_center_x is None:
            return

        if self.mode != "FOLLOW" or self.latest_target is None:
            self.reset_pid()
            self.cmd_vel_pub.publish(twist)
            return

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        self.prev_time = now

        if dt <= 0:
            return

        # Timeout safety
        dt_target = (now - self.last_target_time).nanoseconds / 1e9
        if dt_target > self.target_timeout or self.latest_target.z <= 0.1:
            self.reset_pid()
            self.cmd_vel_pub.publish(twist)
            return

        # === 1. APPLY LOW-PASS FILTER TO SENSOR DATA ===
        if self.smoothed_x is None:
            self.smoothed_x = self.latest_target.x
            self.smoothed_z = self.latest_target.z
        else:
            self.smoothed_x = (self.alpha * self.latest_target.x) + ((1.0 - self.alpha) * self.smoothed_x)
            self.smoothed_z = (self.alpha * self.latest_target.z) + ((1.0 - self.alpha) * self.smoothed_z)

        # === ANGULAR PID ===
        error_x = (self.smoothed_x - self.camera_center_x) / self.camera_center_x

        self.integral_ang += error_x * dt
        derivative_ang = (error_x - self.prev_error_ang) / dt

        angular = (
            self.kp_ang * error_x +
            self.ki_ang * self.integral_ang +
            self.kd_ang * derivative_ang
        )
        self.prev_error_ang = error_x

        # === LINEAR PID ===
        # Use the smoothed Z distance instead of raw
        error_dist = self.smoothed_z - 0.7

        self.integral_lin += error_dist * dt
        derivative_lin = (error_dist - self.prev_error_lin) / dt

        linear = (
            self.kp_lin * error_dist +
            self.ki_lin * self.integral_lin +
            self.kd_lin * derivative_lin
        )
        self.prev_error_lin = error_dist

        # === 2. SOFTEN THE ALIGNMENT FACTOR ===
        # Instead of violently killing speed when off-center, we only slightly reduce it, 
        # and we use the smoothed error_x.
        alignment_factor = max(0.5, 1.0 - abs(error_x)) 
        linear *= alignment_factor

        # Clamp outputs
        twist.linear.x = max(min(linear, 0.5), -0.2)
        twist.angular.z = max(min(-angular, 0.8), -0.8)

        # === OBSTACLE AVOIDANCE ===
        if self.min_center < self.avoid_distance:
            twist.linear.x = min(twist.linear.x, 0.0)
        elif self.min_left < self.avoid_distance:
            twist.linear.x *= 0.5
            twist.angular.z -= 0.6
        elif self.min_right < self.avoid_distance:
            twist.linear.x *= 0.5
            twist.angular.z += 0.6

        self.cmd_vel_pub.publish(twist)

    def reset_pid(self):
        self.integral_ang = 0.0
        self.prev_error_ang = 0.0
        self.integral_lin = 0.0
        self.prev_error_lin = 0.0
        # Reset the filter state so it doesn't drag from an old target
        self.smoothed_x = None 
        self.smoothed_z = None

    def color_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = cv_image.shape
        self.camera_center_x = w // 2

        if self.mode != "FOLLOW":
            return

        if self.latest_target and self.latest_target.z > 0.1:
            cX = int(self.latest_target.x)
            cY = int(self.latest_target.y)

            cv2.circle(cv_image, (cX, cY), 10, (0, 255, 0), -1)
            cv2.putText(cv_image,
                        f"{self.latest_target.z:.2f}m",
                        (cX, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

        cv2.imshow("Follow", cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FollowMeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()