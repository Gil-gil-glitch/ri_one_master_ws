import json
import time
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# インポートパスを絶対パスに修正
from receptionist_system.core.vision import VisionProcessor

# realsense camera 起動コマンド
# ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true         

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()
        
        # 1. パラメータ設定
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.25)
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        conf_thresh = self.get_parameter('conf_threshold').get_parameter_value().double_value
        
        # 2. Visionエンジンの初期化
        self.vision = VisionProcessor(
            model_path=model_path,
            conf_threshold=conf_thresh
        )
        
        # 3. パブリッシャー (解析結果を送信)
        self.detection_pub = self.create_publisher(String, '/receptionist/detections', 10)
        
        # 4. サブスクライバー
        # カラー画像 (YOLO用)
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        # 深度画像 (距離計測用)
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

        self.current_depth_frame = None
        self.get_logger().info('Vision Node with Depth sensing started.')

    def depth_callback(self, msg):
        """深度画像を受け取って保持する"""
        try:
            # 深度画像は16bit(ミリメートル単位)のデータとして受け取る
            self.current_depth_frame = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def image_callback(self, msg):
        """カラー画像を受け取ってYOLO推論と距離計測を行うメインロジック"""
        try:
            color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # YOLO推論実行
            results = self.vision.model(color_frame, conf=self.vision.conf_threshold, device='cuda')[0]
            
            people = []
            chairs = []
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                b = box.xyxy[0].cpu().numpy().astype(int)
                
                if cls == 0:    # Person
                    # --- 距離計測ロジック ---
                    dist_m = 0.0
                    if self.current_depth_frame is not None:
                        # バウンディングボックスの中心点を計算
                        cx = int((b[0] + b[2]) / 2)
                        cy = int((b[1] + b[3]) / 2)
                        
                        # 中心点付近（5x5ピクセル）の平均距離を取得して安定化させる
                        # 0（欠損値）を除外して平均を取る
                        roi = self.current_depth_frame[max(0, cy-2):min(480, cy+2), 
                                                       max(0, cx-2):min(640, cx+2)]
                        valid_pixels = roi[roi > 0]
                        if len(valid_pixels) > 0:
                            dist_m = float(np.mean(valid_pixels)) / 1000.0 # mm -> m
                    
                    people.append({
                        'bbox': b.tolist(), 
                        'conf': conf, 
                        'distance': dist_m
                    })
                    
                elif cls == 56: # Chair
                    chairs.append({'bbox': b.tolist(), 'conf': conf})

            # 状態判定
            status = "searching"
            if len(people) > 0:
                status = "person_detected"

            # 空席判定 (IoU)
            empty_seats = []
            for c in chairs:
                occupied = False
                for p in people:
                    if self.vision._compute_iou(np.array(c['bbox']), np.array(p['bbox'])) > 0.1:
                        occupied = True
                        break
                if not occupied:
                    empty_seats.append(c['bbox'])

            # JSON結果のパブリッシュ
            res = {
                "status": status,
                "people": people,
                "empty_seats": empty_seats
            }
            self.detection_pub.publish(String(data=json.dumps(res)))

            # デバッグ表示
            self._show_debug(color_frame, people, chairs, empty_seats)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

    def _show_debug(self, frame, people, chairs, empty_seats):
        # 人の描画（距離を表示）
        for p in people:
            bbox = p['bbox']
            dist = p['distance']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"Person: {dist:.2f}m", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 椅子の描画
        for s in empty_seats:
            cv2.rectangle(frame, (s[0], s[1]), (s[2], s[3]), (0, 255, 0), 3)
            cv2.putText(frame, "EMPTY", (s[0], s[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Receptionist Vision (Depth Enabled)", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()