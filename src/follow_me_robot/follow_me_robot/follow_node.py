#
#
#  follow_me
#
#  This iteration of follow_me introduces a state machine with the following phases: FOLLOW and RETURN.
#  The state FOLLOW is where the robot is guided by the user to a bag dropoff location, whereas the 
#  the state RETURN is where the robot goes back home to its original starting locaton. The respective
#  signals for these states is pointing and open_palm from the /gesture topic. The CML will use 
#  different signals for these phases, however, these can be easily changed.
#

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, Point, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry  # Added for capturing Home Position
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np

# Nav2 Action
from nav2_msgs.action import NavigateToPose

class FollowMeNode(Node):
    def __init__(self):
        super().__init__('follow_me_node')
        
        self.bridge = CvBridge()
        self.get_logger().info("Follow node active. Mode: IDLE")
        
        # --- State Machine ---
        self.mode = "IDLE" 
        self.home_pose = None # Will store the starting location
        
        # --- Publishers  ----
        self.cmd_vel_pub = self.create_publisher(Twist, '/commands/velocity', 10)

        # --- Subscribers ---
        self.target_sub = self.create_subscription(Point, '/target_person', self.target_callback, 10)
        self.gesture_sub = self.create_subscription(String, '/gesture', self.gesture_callback, 10)
        
        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        
        # Subscribe to odometry to capture initial home location
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # --- Nav2 Action Client ---
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        self.latest_target = None
        self.min_left = 10.0
        self.min_center = 10.0
        self.min_right = 10.0
        
        self.avoid_distance = 0.35 

    def odom_callback(self, msg):
        # Capture the very first odometry reading as our Home Position
        if self.home_pose is None:
            self.home_pose = msg.pose.pose
            self.get_logger().info(f"Home position dynamically recorded: X={self.home_pose.position.x:.2f}, Y={self.home_pose.position.y:.2f}")

    def gesture_callback(self, msg):

        if self.mode == "RETURN":
            self.get_logger().info("Currently returning home. Ignoring gesture commands until return is complete.")
            return
        
        if msg.data == "pointing" and self.mode != "FOLLOW":
            self.get_logger().info("Gesture 'pointing' received. Switching to FOLLOW mode.")
            self.mode = "FOLLOW"
            
        elif msg.data == "open_palm" and self.mode != "RETURN":
            self.get_logger().info("Gesture 'open_palm' received. Halting robot and returning home.")
            self.mode = "RETURN"
            
            # FIX: Explicitly stop the robot before handing over to Nav2!
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            
            self.send_return_home_goal()

    def send_return_home_goal(self):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 server not available! Cannot return home.")
            return

        if self.home_pose is None:
            self.get_logger().error("Home position was never recorded. Cannot return.")
            return

        goal_msg = NavigateToPose.Goal()
        
        # Change 'odom' to 'map' if you are using AMCL/SLAM for the Navigation Phase
        goal_msg.pose.header.frame_id = 'odom'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Use the dynamically recorded home pose
        goal_msg.pose.pose = self.home_pose

        self.get_logger().info("Sending dynamic home coordinates to Nav2...")
        self.nav_client.send_goal_async(goal_msg)

    def scan_callback(self, msg):
        # FIX: Do not treat objects closer than 0.25m as 10.0m! 
        # Only ignore 0.0 (invalid readings) and filter out parts of the robot chassis itself (e.g. < 0.10m)
        ranges = np.array([r if r > 0.10 and r < 10.0 else 10.0 for r in msg.ranges])
        num_points = len(ranges)
        
        if num_points == 0:
            return

        # FIX: Widened the LIDAR cones to 45 degrees to see more of the environment
        deg_45 = int(num_points * (45.0 / 360.0))
        deg_15 = int(num_points * (15.0 / 360.0))
        
        left_arc = ranges[deg_15 : deg_45]
        center_arc = np.concatenate((ranges[:deg_15], ranges[-deg_15:]))
        right_arc = ranges[-deg_45 : -deg_15]
        
        self.min_left = np.min(left_arc) if len(left_arc) > 0 else 10.0
        self.min_center = np.min(center_arc) if len(center_arc) > 0 else 10.0
        self.min_right = np.min(right_arc) if len(right_arc) > 0 else 10.0

    def target_callback(self, msg):
        self.latest_target = msg

    def color_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        height, width, _ = cv_image.shape
        center_x = width // 2

        if self.mode == "FOLLOW":
            twist = Twist()
            
            if self.latest_target is not None and self.latest_target.z > 0.1:
                cX = int(self.latest_target.x)
                cY = int(self.latest_target.y)
                distance_meters = self.latest_target.z
                
                error_x = center_x - cX
                twist.angular.z = float(error_x * 0.002) 

                distance_error = distance_meters - 0.7 
                if abs(distance_error) > 0.02: 
                    twist.linear.x = float(distance_error * 0.5) 
                
                # Obstacle Avoidance
                if self.min_center < self.avoid_distance:
                    if twist.linear.x > 0.0:
                        twist.linear.x = 0.0 
                elif self.min_left < self.avoid_distance:
                    if twist.linear.x > 0.0:
                        twist.linear.x *= 0.5  
                    twist.angular.z -= 0.6 
                elif self.min_right < self.avoid_distance:
                    if twist.linear.x > 0.0:
                        twist.linear.x *= 0.5  
                    twist.angular.z += 0.6 
                    
                twist.linear.x = max(min(twist.linear.x, 0.6), -0.4) 
                twist.angular.z = max(min(twist.angular.z, 1.0), -1.0) 
                
                self.cmd_vel_pub.publish(twist)

        elif self.mode == "RETURN":
            cv2.putText(cv_image, "RETURNING TO BASE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Follow Me Camera", cv_image)
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