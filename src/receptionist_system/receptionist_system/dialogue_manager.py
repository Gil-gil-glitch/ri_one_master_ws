import json
import rclpy
import math
import numpy as np

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

class DialogueManager(Node):
    def __init__(self):
        super().__init__('dialogue_manager')

        # --- 【現場調整が必要なパラメータ】 ---
        # 1. 部屋の境界（Map座標系 / メートル単位）
        self.ROOM_X_MIN = 0
        self.ROOM_X_MAX = 1.05
        self.ROOM_Y_MIN = -1.85
        self.ROOM_Y_MAX = -1.50

        # 2. 目標地点の座標（Map座標系）
        # SEAT_DEFAULTは「2つの椅子の中間地点」など、椅子があるエリアを向くための座標にします。
        self.poses = {
           'DOOR': {  'x': 0.12514515221118927,  'y': -1.8234933614730835, 'z': 0.002471923828125, 'w': 1.0 },
           'HOST': {'x': 1.0387544631958008, 'y': -1.673363447189331, 'z': 0.002471923828125, 'w': 1.0},
           'SEAT_DEFAULT': {'x': 0.7947089672088623, 'y': -1.5169737339019775, 'z':  0.002471923828125, 'w': 1.0}
        }
        # ------------------------------------

        # 通信設定
        self.sub_action = self.create_subscription(String, '/task_action', self.action_cb, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_cb, 10)
        self.create_subscription(String, '/receptionist/detections', self.detections_cb, 10)
        
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)
        self.pub_status = self.create_publisher(String, '/action_status', 10) 
        
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        
        self.latest_depth = None
        self.camera_info = None
        self.current_guest = {}
        self.last_empty_seat_pixel = None

    def euler_yaw_to_quat(self, yaw):
        return {'x': 0.0, 'y': 0.0, 'z': math.sin(yaw / 2.0), 'w': math.cos(yaw / 2.0)}

    # ======== コールバック処理 ========
    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, msg.encoding)

    def camera_info_cb(self, msg):
        self.camera_info = msg

    def detections_cb(self, msg):
        """VisionNodeから空席情報を受け取る"""
        data = json.loads(msg.data)
        seats = data.get("empty_seats", [])
        if seats:
            # 2つの椅子のうち、Visionが見つけた「空いている方」の1つ目を記録
            s = seats[0]
            self.last_empty_seat_pixel = [(s[0]+s[2])/2, (s[1]+s[3])/2]
        else:
            self.last_empty_seat_pixel = None

    # ======== 座標計算ロジック ========
    def get_map_coords_from_pixel(self, u, v):
        if self.latest_depth is None or self.camera_info is None:
            return None
        try:
            depth = self.latest_depth[int(v), int(u)] / 1000.0
            if depth < 0.3 or depth > 5.0: return None

            fx, fy = self.camera_info.k[0], self.camera_info.k[4]
            cx, cy = self.camera_info.k[2], self.camera_info.k[5]

            point_cam = PointStamped()
            point_cam.header.frame_id = self.camera_info.header.frame_id
            point_cam.header.stamp = self.get_clock().now().to_msg()
            point_cam.point.x = (u - cx) * depth / fx
            point_cam.point.y = (v - cy) * depth / fy
            point_cam.point.z = depth

            return self.tf_buffer.transform(point_cam, 'map', timeout=rclpy.duration.Duration(seconds=1.0)).point
        except:
            return None

    def is_inside_room(self, pt):
        """座標が設定した部屋の範囲内にあるかチェック"""
        if pt is None: return False
        return (self.ROOM_X_MIN < pt.x < self.ROOM_X_MAX) and \
               (self.ROOM_Y_MIN < pt.y < self.ROOM_Y_MAX)

    def get_quaternion_to_face(self, target_x, target_y):
        """現在の位置からターゲットの方を向くためのクォータニオンを計算"""
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time()).transform.translation
            yaw = math.atan2(target_y - t.y, target_x - t.x)
            return self.euler_yaw_to_quat(yaw)
        except:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}

    # ======== ナビゲーション・アクション送信 ========
    def action_cb(self, msg):
        req = json.loads(msg.data)
        action = req.get("action")
        
        if action == "APPROACH_GUEST":
            bbox = req.get("data", {}).get("bbox", [])
            if len(bbox) == 4:
                target_pt = self.get_map_coords_from_pixel((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
                if target_pt:
                    t = self.tf_buffer.lookup_transform('map', 'base_link', Time()).transform.translation
                    dist = math.sqrt((target_pt.x - t.x)**2 + (target_pt.y - t.y)**2)
                    ratio = (dist - 1.0) / dist if dist > 1.0 else 0.1
                    self.send_custom_goal(t.x + (target_pt.x - t.x) * ratio, t.y + (target_pt.y - t.y) * ratio, 
                                          self.get_quaternion_to_face(target_pt.x, target_pt.y), "GUEST_NEAR")
                    return
            self.send_navigation_goal('DOOR')

        elif action == "MOVE_TO_HOST":
            self.current_guest = req.get("data", {})
            self.say(f"Follow me, {self.current_guest.get('name')}. I will take you to Cris.")
            self.send_navigation_goal('HOST')
            
        elif action == "MOVE_TO_DOOR":
            self.send_navigation_goal('DOOR')

    def send_navigation_goal(self, label):
        pose = self.poses.get(label, self.poses['DOOR'])
        self.send_custom_goal(pose['x'], pose['y'], pose, label)

    def send_custom_goal(self, x, y, quat, label):
        if not self.nav_client.wait_for_server(timeout_sec=2.0): return
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = quat['z']
        goal_msg.pose.pose.orientation.w = quat['w']
        
        self.nav_client.send_goal_async(goal_msg).add_done_callback(
            lambda f: f.result().get_result_async().add_done_callback(
                lambda f2: self.custom_result_callback(f2, label)))

    # ======== シーケンス制御 (重要) ========
    def custom_result_callback(self, future, label):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            if label == "GUEST_NEAR":
                self.pub_status.publish(String(data="ARRIVED_AT_GUEST"))
            
            elif label == "HOST":
                # ホストの位置に到着。次はホストの方を向く。
                self.start_rotation_to_target("HOST_FACE", self.poses['HOST']['x'], self.poses['HOST']['y'])
            
            elif label == "HOST_FACE":
                # ホストを向いたので挨拶。次に「椅子エリア」の方を向く。
                self.introduce_to_host()
                self.start_rotation_to_target("SCAN_FOR_SEAT", self.poses['SEAT_DEFAULT']['x'], self.poses['SEAT_DEFAULT']['y'])
            
            elif label == "SCAN_FOR_SEAT":
                # 椅子エリアを向いた状態。ここで最新のVision結果を見て、正確な椅子を狙う。
                self.point_to_actual_seat()
            
            elif label == "SEAT_FACE":
                # 正確な椅子を向いたので、指を差して完了。
                self.point_and_finish()
            
            elif label == "DOOR":
                self.pub_status.publish(String(data="ARRIVED_AT_DOOR"))
        else:
            self.get_logger().warn(f"Action {label} failed")

    def start_rotation_to_target(self, next_label, tx, ty):
        """今いる場所で向きだけ変える"""
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time()).transform.translation
            quat = self.get_quaternion_to_face(tx, ty)
            self.send_custom_goal(t.x, t.y, quat, next_label)
        except:
            self.custom_result_callback(None, next_label)

    def introduce_to_host(self):
        self.say(f"Hello Cris! I have brought {self.current_guest.get('name')}.")
        self.say(f"Their favorite drink is {self.current_guest.get('drink')}.")

    def point_to_actual_seat(self):
        """カメラの視界に入った実際の空席を確認し、微調整する"""
        tx, ty = self.poses['SEAT_DEFAULT']['x'], self.poses['SEAT_DEFAULT']['y']
        
        if self.last_empty_seat_pixel:
            seat_pt = self.get_map_coords_from_pixel(self.last_empty_seat_pixel[0], self.last_empty_seat_pixel[1])
            # 部屋の範囲内（ホストの椅子エリア以外など）にあれば採用
            if self.is_inside_room(seat_pt):
                tx, ty = seat_pt.x, seat_pt.y
                self.get_logger().info(f"Dynamic seat detected: {tx}, {ty}")

        # 最終的な椅子（またはデフォルト）に向かって旋回
        self.start_rotation_to_target("SEAT_FACE", tx, ty)

    def point_and_finish(self):
        self.say("There is an empty seat for you. Please take a seat.")
        self.move_arm([0.0, 0.4, 0.2, 0.0]) # 指差し
        self.pub_status.publish(String(data="COMPLETED_GUEST_MANAGEMENT"))

    # ======== アーム・音声出力 ========
    def move_arm(self, pos):
        if not self.arm_client.wait_for_server(timeout_sec=1.0): return
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start.sec = 2
        goal_msg.trajectory.points = [point]
        self.arm_client.send_goal_async(goal_msg)

    def say(self, text):
        self.pub_tts.publish(String(data=text))

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(DialogueManager())
    rclpy.shutdown()
