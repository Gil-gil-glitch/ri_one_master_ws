import json
import rclpy
import math
import numpy as np
import sys
import termios
import tty
import threading
import time

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
        self.cmd_vel_pub = self.create_publisher(Twist, '/commands/velocity', 1)

        # Action Clients
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # =====================================================
        # ★ マッピング後にここの座標を実測値に変更してください
        #   ros2 topic echo /amcl_pose --once で取得
        # =====================================================
        self.poses = {
            'DOOR': {
                'x': 1.9020752906799316, 'y': -0.5670076012611389, 'yaw': 0.0,
            },
            'HOST': {
                'x': 1.5482778549194336, 'y': -3.1809372901916504,
                'yaw': 0.0,
            },
            'SEAT_DEFAULT': {
                'x': 0.26736369729042053, 'y':  -4.002254486083984,
                'yaw': 0.0,
            },
            'JUDGE1': {
                'x': 0.7906104922294617, 'y': -3.1809372901916504, # Chris
                'yaw': 0.0,
            },
            'JUDGE2': {
                'x': 0.7906104922294617, 'y':  -3.235915184020996, # Assume Guest 1
                'yaw': 0.0,
            },
        }

        # ★ 前進距離[m]
        self.FORWARD_DISTANCE = 1.2

        # ★ 前進速度[m/s]
        self.FORWARD_SPEED = 0.15

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        self.latest_depth = None
        self.camera_info = None
        self.current_guest = {}
        self.last_empty_seat_pixel = None

        self._current_goal_handle = None
        self._emergency_stopped = False

        # 緊急停止スレッド
        self._kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._kb_thread.start()

        self.get_logger().info("DialogueManager started. Press 'q' or SPACE to emergency-stop.")

    # ======================================================
    # Emergency Stop
    # ======================================================
    def _keyboard_listener(self):
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
        self._emergency_stopped = True
        if self._current_goal_handle is not None:
            try:
                self._current_goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().error(f"Goal cancel error: {e}")
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().warn("Robot stopped.")

    # ======================================================
    # Callbacks
    # ======================================================
    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, msg.encoding)

    def camera_info_cb(self, msg):
        self.camera_info = msg

    def detections_cb(self, msg):
        data = json.loads(msg.data)
        seats = data.get("empty_seats", [])
        if seats:
            s = seats[0]
            self.last_empty_seat_pixel = [(s[0] + s[2]) / 2, (s[1] + s[3]) / 2]

    # ======================================================
    # Utility
    # ======================================================
    def euler_yaw_to_quat(self, yaw: float) -> dict:
        return {
            'x': 0.0,
            'y': 0.0,
            'z': math.sin(yaw / 2.0),
            'w': math.cos(yaw / 2.0),
        }

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

    def get_quaternion_to_face(self, target_x, target_y) -> dict:
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time()).transform.translation
            yaw = math.atan2(target_y - t.y, target_x - t.x)
            return self.euler_yaw_to_quat(yaw)
        except Exception:
            return self.euler_yaw_to_quat(0.0)

    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time()).transform
            return t.translation, t.rotation
        except Exception:
            return None, None

    # ======================================================
    # Action callback
    # ======================================================
    def action_cb(self, msg):
        if self._emergency_stopped:
            self.get_logger().warn("Emergency stop active. Ignoring action.")
            return

        req = json.loads(msg.data)
        action = req.get("action")

        if action == "MOVE_FORWARD_TO_GUEST":
            self._move_forward(self.FORWARD_DISTANCE)

        elif action == "MOVE_TO_HOST":
            self.current_guest = req.get("data", {})
            self.say(f"Follow me, {self.current_guest.get('name')}. I will take you to Chris.")
            self._send_nav_goal_from_pose('HOST')

        elif action == "MOVE_TO_DOOR":
            self._send_nav_goal_from_pose('DOOR')

        elif action == "MOVE_TO_FACE_JUDGE_1":
            self._send_nav_goal_from_pose('JUDGE1')

        elif action == "MOVE_TO_FACE_JUDGE_2":
            self._send_nav_goal_from_pose('JUDGE2')

        elif action == "SAY_TEXT":
            text = req.get("text", "")
            self.say(text)
            threading.Timer(5.0, lambda: self.pub_status.publish(String(data="BONUS_TEXT_DONE"))).start()

    # ======================================================
    # ★ 前進動作（cmd_velで直接制御）
    # ======================================================
    def _move_forward(self, distance: float):
        self.get_logger().info(f"Moving forward {distance:.2f}m...")
        speed = self.FORWARD_SPEED
        duration = distance / speed

        def _forward_thread():
            twist = Twist()
            twist.linear.x = speed
            rate_hz = 20
            steps = int(duration * rate_hz)

            for _ in range(steps):
                if self._emergency_stopped:
                    break
                self.cmd_vel_pub.publish(twist)
                time.sleep(1.0 / rate_hz)

            stop_twist = Twist()
            for _ in range(10):
                self.cmd_vel_pub.publish(stop_twist)
                time.sleep(0.05)

            self.get_logger().info("Forward movement done.")
            self.pub_status.publish(String(data="ARRIVED_AT_GUEST"))

        threading.Thread(target=_forward_thread, daemon=True).start()

    # ======================================================
    # Navigation
    # ======================================================
    def _send_nav_goal_from_pose(self, label: str):
        pose = self.poses.get(label)
        if pose is None:
            self.get_logger().error(f"Unknown pose label: {label}")
            return
        quat = self.euler_yaw_to_quat(pose['yaw'])
        self._send_custom_goal(pose['x'], pose['y'], quat, label)

    def _send_custom_goal(self, x: float, y: float, quat: dict, label: str):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 server not available.")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = quat['z']
        goal_msg.pose.pose.orientation.w = quat['w']

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self._on_goal_accepted(f, label))

    def _on_goal_accepted(self, future, label: str):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Goal '{label}' rejected by Nav2.")
            return
        self._current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            lambda f: self._on_goal_result(f, label)
        )

    def _on_goal_result(self, future, label: str):
        if self._emergency_stopped:
            return
        try:
            status = future.result().status
        except Exception:
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            if label == "HOST":
                self._start_rotation("HOST_FACE", self.poses['HOST']['x'], self.poses['HOST']['y'])
            elif label == "HOST_FACE":
                self._introduce_and_rotate_to_seat()
            elif label == "SEAT_FACE":
                self._point_and_finish()
            elif label == "DOOR":
                self.pub_status.publish(String(data="ARRIVED_AT_DOOR"))
            elif label == "JUDGE1":
                self.pub_status.publish(String(data="FACING_JUDGE_1"))
            elif label == "JUDGE2":
                self.pub_status.publish(String(data="FACING_JUDGE_2"))
        else:
            self.get_logger().warn(f"Action '{label}' failed. Executing fallback.")
            if label == "HOST":
                self._start_rotation("HOST_FACE", self.poses['HOST']['x'], self.poses['HOST']['y'])
            elif label == "HOST_FACE":
                self._introduce_and_rotate_to_seat()
            elif label == "SEAT_FACE":
                self._point_and_finish()
            elif label == "DOOR":
                self.pub_status.publish(String(data="ARRIVED_AT_DOOR"))
            elif label == "JUDGE1":
                self.pub_status.publish(String(data="FACING_JUDGE_1"))
            elif label == "JUDGE2":
                self.pub_status.publish(String(data="FACING_JUDGE_2"))

    def _start_rotation(self, next_label: str, tx: float, ty: float):
        try:
            t_data = self.tf_buffer.lookup_transform('map', 'base_link', Time()).transform.translation
            quat = self.get_quaternion_to_face(tx, ty)
            self._send_custom_goal(t_data.x, t_data.y, quat, next_label)
        except Exception as e:
            self.get_logger().error(f"Rotation error: {e}")
            self.pub_status.publish(String(data="ARRIVED_AT_DOOR")) # Fallback signal

    def _introduce_and_rotate_to_seat(self):
        name = self.current_guest.get('name', '?')
        drink = self.current_guest.get('drink', '?')
        self.say(f"Hello Chris! I have brought {name}. Their favorite drink is {drink}.")
        tx, ty = self.poses['SEAT_DEFAULT']['x'], self.poses['SEAT_DEFAULT']['y']
        if self.last_empty_seat_pixel:
            seat_pt = self.get_map_coords_from_pixel(self.last_empty_seat_pixel[0], self.last_empty_seat_pixel[1])
            if seat_pt: tx, ty = seat_pt.x, seat_pt.y
        self._start_rotation("SEAT_FACE", tx, ty)

    def _point_and_finish(self):
        self.say("There is an empty seat for you. Please take a seat.")
        self.move_arm([0.0, 0.4, 0.2, 0.0])
        self.pub_status.publish(String(data="COMPLETED_GUEST_MANAGEMENT"))

    def move_arm(self, pos):
        if not self.arm_client.wait_for_server(timeout_sec=1.0): return
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        point = JointTrajectoryPoint()
        point.positions, point.time_from_start.sec = pos, 2
        goal_msg.trajectory.points = [point]
        self.arm_client.send_goal_async(goal_msg)

    def say(self, text: str):
        self.pub_tts.publish(String(data=text))

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(DialogueManager())
    rclpy.shutdown()

if __name__ == '__main__':
    main()