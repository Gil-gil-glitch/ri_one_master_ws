import json
import rclpy
import math
import numpy as np
import sys
import termios
import tty
import threading

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Twist
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class DialogueManager(Node):
    def __init__(self):
        super().__init__('dialogue_manager')

        # Subscribers
        self.sub_action = self.create_subscription(String, '/task_action', self.action_cb, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_cb, 10)
        self.create_subscription(String, '/receptionist/detections', self.detections_cb, 10)

        # Publishers
        self.pub_tts = self.create_publisher(String, '/robot_speech', 10)
        self.pub_status = self.create_publisher(String, '/action_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # Action Clients
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # =====================================================
        # ★ マッピング後にここの座標を実測値に変更してください ★
        # 取得方法: ロボットを各地点に手動移動後、以下を実行
        #   ros2 topic echo /amcl_pose --once
        # =====================================================
        self.poses = {
            'DOOR':  {  'x': 1.989912748336792, 'y': -0.5786458253860474,  'z': -0.001434326171875, 'w': 1.0},
            'HOST':   {   'x': 0.9125170707702637, 'y': -3.2827506065368652, 'z': 0.0025634765625, 'w': 1.0},
            'SEAT_DEFAULT': {  'x': 0.3470032513141632, 'y': -4.002254486083984, 'z': 0.0025634765625, 'w': 1.0},
        }

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        self.latest_depth = None
        self.camera_info = None
        self.current_guest = {}
        self.last_empty_seat_pixel = None

        # ── goal handle を保存してキャンセルできるようにする ──
        self._current_goal_handle = None
        self._emergency_stopped = False

        # ── キーボード緊急停止スレッド ──
        self._kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._kb_thread.start()
        self.get_logger().info("DialogueManager started.  Press 'q' or SPACE to emergency-stop.")

    # ======================================================
    # Emergency Stop
    # ======================================================
    def _keyboard_listener(self):
        """q / Q / スペースで緊急停止"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while rclpy.ok():
                ch = sys.stdin.read(1)
                if ch in ('q', 'Q', ' '):
                    self.get_logger().warn("=== EMERGENCY STOP TRIGGERED ===")
                    self._emergency_stop()
        except Exception as e:
            self.get_logger().error(f"Keyboard listener error: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _emergency_stop(self):
        """Nav2 ゴールをキャンセルし /cmd_vel でゼロ速度を送信して即停止"""
        self._emergency_stopped = True
        # goal handle が保存されていればキャンセル
        if self._current_goal_handle is not None:
            try:
                self._current_goal_handle.cancel_goal_async()
                self.get_logger().warn("Navigation goal cancelled.")
            except Exception as e:
                self.get_logger().error(f"Goal cancel error: {e}")
        # /cmd_vel にゼロを送って確実に停止
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().warn("Robot stopped. Fix the issue and restart the node.")

    # ======================================================
    # Callbacks
    # ======================================================
    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, msg.encoding)

    def camera_info_cb(self, msg):
        self.camera_info = msg

    def detections_cb(self, msg):
        """VisionNode からの情報を受け取り、空いている椅子を記憶する"""
        data = json.loads(msg.data)
        seats = data.get("empty_seats", [])
        if seats:
            s = seats[0]
            self.last_empty_seat_pixel = [(s[0] + s[2]) / 2, (s[1] + s[3]) / 2]

    # ======================================================
    # 3D Coordinate / Rotation Utils
    # ======================================================
    def get_map_coords_from_pixel(self, u, v):
        if self.latest_depth is None or self.camera_info is None:
            return None
        try:
            depth = self.latest_depth[int(v), int(u)] / 1000.0
            if depth == 0:
                return None

            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]

            point_cam = PointStamped()
            point_cam.header.frame_id = self.camera_info.header.frame_id
            point_cam.header.stamp = self.get_clock().now().to_msg()
            point_cam.point.x = (u - cx) * depth / fx
            point_cam.point.y = (v - cy) * depth / fy
            point_cam.point.z = depth

            return self.tf_buffer.transform(
                point_cam, 'map',
                timeout=rclpy.duration.Duration(seconds=1.0)
            ).point
        except Exception:
            return None

    def euler_yaw_to_quat(self, yaw):
        return {
            'x': 0.0,
            'y': 0.0,
            'z': math.sin(yaw / 2.0),
            'w': math.cos(yaw / 2.0),
        }

    def get_quaternion_to_face(self, target_x, target_y):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', Time()
            ).transform.translation
            yaw = math.atan2(target_y - t.y, target_x - t.x)
            return self.euler_yaw_to_quat(yaw)
        except Exception:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}

    # ======================================================
    # Navigation
    # ======================================================
    def action_cb(self, msg):
        if self._emergency_stopped:
            self.get_logger().warn("Emergency stop is active. Ignoring action.")
            return

        req = json.loads(msg.data)
        action = req.get("action")

        if action == "APPROACH_GUEST":
            bbox = req.get("data", {}).get("bbox", [])
            if len(bbox) == 4:
                u = (bbox[0] + bbox[2]) / 2
                v = (bbox[1] + bbox[3]) / 2
                target_pt = self.get_map_coords_from_pixel(u, v)
                if target_pt:
                    try:
                        t = self.tf_buffer.lookup_transform(
                            'map', 'base_link', Time()
                        ).transform.translation
                        dist = math.sqrt(
                            (target_pt.x - t.x) ** 2 + (target_pt.y - t.y) ** 2
                        )
                        if dist < 1.5:
                            self.get_logger().info(
                                f"Guest is already close ({dist:.2f}m). Starting reception."
                            )
                            self.pub_status.publish(String(data="ARRIVED_AT_GUEST"))
                            return

                        ratio = (dist - 1.0) / dist
                        target_x = t.x + (target_pt.x - t.x) * ratio
                        target_y = t.y + (target_pt.y - t.y) * ratio
                        self.get_logger().info(f"Approaching guest. Distance: {dist:.2f}m")
                        self.send_custom_goal(
                            target_x, target_y,
                            self.get_quaternion_to_face(target_pt.x, target_pt.y),
                            "GUEST_NEAR"
                        )
                        return
                    except Exception as e:
                        self.get_logger().error(f"Transform error: {e}")

            # フォールバック: 座標不明の場合はドアへ
            self.send_navigation_goal('DOOR')

        elif action == "MOVE_TO_HOST":
            self.current_guest = req.get("data", {})
            self.say(f"Follow me, {self.current_guest.get('name')}. I will take you to Chris.")
            self.send_navigation_goal('HOST')

        elif action == "MOVE_TO_DOOR":
            self.send_navigation_goal('DOOR')

    def send_navigation_goal(self, label):
        pose = self.poses.get(label, self.poses['DOOR'])
        self.send_custom_goal(pose['x'], pose['y'], pose, label)

    def send_custom_goal(self, x, y, quat, label):
        """Nav2 へ目標を送信し、goal handle を self._current_goal_handle に保存する"""
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 server not available.")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = quat['z']
        goal_msg.pose.pose.orientation.w = quat['w']

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda f: self._on_goal_accepted(f, label)
        )

    def _on_goal_accepted(self, future, label):
        """goal handle を保存してから result を待つ"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Goal '{label}' was rejected by Nav2.")
            return
        # ★ goal handle を保存 → 緊急停止時にキャンセル可能
        self._current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self.custom_result_callback(f, label)
        )

    def custom_result_callback(self, future, label):
        """移動・旋回完了後の状態遷移"""
        # 緊急停止中は何もしない
        if self._emergency_stopped:
            return

        try:
            status = future.result().status
        except Exception:
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            if label == "GUEST_NEAR":
                self.pub_status.publish(String(data="ARRIVED_AT_GUEST"))
            elif label == "HOST":
                self.start_rotation_to_target(
                    "HOST_FACE",
                    self.poses['HOST']['x'],
                    self.poses['HOST']['y']
                )
            elif label == "HOST_FACE":
                self.introduce_and_rotate_to_seat()
            elif label == "SEAT_FACE":
                self.point_and_finish()
            elif label == "DOOR":
                self.pub_status.publish(String(data="ARRIVED_AT_DOOR"))
        else:
            self.get_logger().warn(f"Action '{label}' failed with status {status}")

    # ======================================================
    # Sequence Logic
    # ======================================================
    def start_rotation_to_target(self, next_label, tx, ty):
        """指定座標を向くための旋回（移動なし）を発行"""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', Time()
            ).transform.translation
            quat = self.get_quaternion_to_face(tx, ty)
            self.send_custom_goal(t.x, t.y, quat, next_label)
        except Exception:
            # 変換失敗時は次のステップへ強制進行
            self.custom_result_callback(
                type('F', (), {'result': lambda s: type('R', (), {'status': GoalStatus.STATUS_SUCCEEDED})()})(),
                next_label
            )

    def introduce_and_rotate_to_seat(self):
        """ホストへ挨拶し、空席へ旋回する"""
        name = self.current_guest.get('name', '?')
        drink = self.current_guest.get('drink', '?')
        self.say(f"Hello Chris! I have brought {name}.")
        self.say(f"Their favorite drink is {drink}.")

        tx, ty = self.poses['SEAT_DEFAULT']['x'], self.poses['SEAT_DEFAULT']['y']
        if self.last_empty_seat_pixel:
            seat_pt = self.get_map_coords_from_pixel(
                self.last_empty_seat_pixel[0],
                self.last_empty_seat_pixel[1]
            )
            if seat_pt:
                tx, ty = seat_pt.x, seat_pt.y

        self.start_rotation_to_target("SEAT_FACE", tx, ty)

    def point_and_finish(self):
        """椅子を指差してタスク完了"""
        self.say("There is an empty seat for you. Please take a seat.")
        self.move_arm([0.0, 0.4, 0.2, 0.0])
        self.pub_status.publish(String(data="COMPLETED_GUEST_MANAGEMENT"))

    # ======================================================
    # Arm & Speech
    # ======================================================
    def move_arm(self, pos):
        if not self.arm_client.wait_for_server(timeout_sec=1.0):
            return
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