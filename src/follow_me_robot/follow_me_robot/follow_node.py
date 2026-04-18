import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, LaserScan
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np

class FollowMeNode(Node):
    def __init__(self):
        super().__init__('follow_me_node')
        
        self.bridge = CvBridge()
        self.get_logger().info("Follow node active. Waiting for target data from targetting node...")
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/commands/velocity', 10)
        
        # Subscribe directly to the point published by person_targetting_node
        self.target_sub = self.create_subscription(Point, '/target_person', self.target_callback, 10)
        
        # We only keep the color sub to display the tracking UI
        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        
        self.latest_target = None
        self.min_left = 10.0
        self.min_center = 10.0
        self.min_right = 10.0
        
        self.target_distance = 0.7     
        self.avoid_distance = 0.3      
        
        self.linear_p_gain = 0.8       
        self.angular_p_gain = 0.003    

    def target_callback(self, msg):
        self.latest_target = msg

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

    def color_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = cv_image.shape
        center_x_screen = width // 2
        twist = Twist() 

        # Only move if we have a valid target (z > 0.1)
        if self.latest_target is not None and self.latest_target.z > 0.1:
            cX = int(self.latest_target.x)
            cY = int(self.latest_target.y)
            distance_meters = self.latest_target.z

            error_x = center_x_screen - cX 
            twist.angular.z = float(error_x * self.angular_p_gain)
            
            distance_error = distance_meters - self.target_distance
            if abs(distance_error) > 0.1: 
                twist.linear.x = float(distance_error * self.linear_p_gain)
                
            if twist.linear.x > 0:
                status_text = "TRACKING"
                color = (0, 255, 0)

                if self.min_center < self.avoid_distance:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.5 if self.min_left > self.min_right else -0.5
                    status_text = "DODGING CENTER"
                    color = (0, 0, 255)
                    
                elif self.min_left < self.avoid_distance:
                    twist.linear.x *= 0.5  
                    twist.angular.z -= 0.6 
                    status_text = "DODGING LEFT"
                    color = (0, 165, 255)
                    
                elif self.min_right < self.avoid_distance:
                    twist.linear.x *= 0.5  
                    twist.angular.z += 0.6 
                    status_text = "DODGING RIGHT"
                    color = (0, 165, 255)

                cv2.putText(cv_image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            twist.linear.x = max(min(twist.linear.x, 0.4), -0.4) 
            twist.angular.z = max(min(twist.angular.z, 1.0), -1.0) 

            cv2.circle(cv_image, (cX, cY), 10, (0, 255, 0), -1) 
            cv2.putText(cv_image, f"{distance_meters:.2f}m", (cX - 20, cY - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Explicitly stop the robot if no valid target exists
            cv2.putText(cv_image, "WAITING FOR TARGET", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

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