import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class FollowMeNode(Node):
    def __init__(self):
        super().__init__('follow_me_node')
        
        self.bridge = CvBridge()
        self.get_logger().info("Loading YOLOv8 AI Model...")
        self.model = YOLO('yolov8n-seg.pt') 
        self.get_logger().info("AI Loaded! Waiting for sensor data...")
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/commands/velocity', 10)
        
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        self.subscription = self.create_subscription(String, "/gesture", self.gesture_callback, 10)
        
        self.latest_depth_img = None
        self.min_left = 10.0
        self.min_center = 10.0
        self.min_right = 10.0
        
        self.target_distance = 0.7     # Follow distance (0/7 meter)
        self.avoid_distance = 0.3      
        
        self.linear_p_gain = 0.8       
        self.angular_p_gain = 0.003    

    def scan_callback(self, msg):
        ranges = np.array([r if 0.1 < r < 10.0 else 10.0 for r in msg.ranges])
        num_points = len(ranges)
        
        if num_points == 0:
            return

        deg_45 = int(num_points * (45.0 / 360.0))
        deg_15 = int(num_points * (15.0 / 360.0))
        
        left_arc = ranges[deg_15 : deg_45]
        center_arc = np.concatenate((ranges[:deg_15], ranges[-deg_15:]))
        right_arc = ranges[-deg_45 : -deg_15]
        
        self.min_left = np.min(left_arc) if len(left_arc) > 0 else 10.0
        self.min_center = np.min(center_arc) if len(center_arc) > 0 else 10.0
        self.min_right = np.min(right_arc) if len(right_arc) > 0 else 10.0

    def depth_callback(self, msg):
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def color_callback(self, msg):
        if self.latest_depth_img is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = cv_image.shape
        center_x_screen = width // 2

        results = self.model(cv_image, classes=[0], verbose=False)
        twist = Twist() 

        if len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy() 
            mask = cv2.resize(mask, (width, height))
            
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]) 
                cY = int(M["m01"] / M["m00"]) 
                
                distance_mm = self.latest_depth_img[cY, cX] 
                distance_meters = distance_mm / 1000.0

                if distance_meters > 0.1:
                    error_x = center_x_screen - cX 
                    twist.angular.z = float(error_x * self.angular_p_gain)
                    
                    distance_error = distance_meters - self.target_distance
                    if abs(distance_error) > 0.1: 
                        twist.linear.x = float(distance_error * self.linear_p_gain)
                        
                    if twist.linear.x > 0:
                        status_text = "TRACKING"
                        color = (0, 255, 0) # Green

                        if self.min_center < self.avoid_distance:
                            twist.linear.x = 0.0
                            if self.min_left > self.min_right:
                                twist.angular.z = 0.5  # Left is more open, turn left
                            else:
                                twist.angular.z = -0.5 # Right is more open, turn right
                            status_text = "DODGING CENTER"
                            color = (0, 0, 255) # Red
                            
                        elif self.min_left < self.avoid_distance:
                            twist.linear.x *= 0.5  
                            twist.angular.z -= 0.6 
                            status_text = "DODGING LEFT"
                            color = (0, 165, 255) # Orange
                            
                        elif self.min_right < self.avoid_distance:
                            twist.linear.x *= 0.5  
                            twist.angular.z += 0.6 
                            status_text = "DODGING RIGHT"
                            color = (0, 165, 255) # Orange

                        # Draw the status on the screen
                        cv2.putText(cv_image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                    twist.linear.x = max(min(twist.linear.x, 0.4), -0.4) 
                    twist.angular.z = max(min(twist.angular.z, 1.0), -1.0) 

                    # Draw targeting dot
                    cv2.circle(cv_image, (cX, cY), 10, (0, 255, 0), -1) 
                    cv2.putText(cv_image, f"{distance_meters:.2f}m", (cX - 20, cY - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.cmd_vel_pub.publish(twist)
        cv2.imshow("Robot View", cv_image)
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

