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
        # カメラ解像度はトピックから受け取るため、ここでは推論用設定のみ
        self.vision = VisionProcessor(
            model_path=model_path,
            conf_threshold=conf_thresh
        )
        
        # 3. パブリッシャー (解析結果を送信)
        self.detection_pub = self.create_publisher(String, '/receptionist/detections', 10)
        
        # 4. サブスクライバー (RealSenseのトピックを購読)
        # 前回の質問で成功した 640x480@30fps の画像がここに入ってきます
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        self.get_logger().info('Vision Node started: Subscribing to /camera/color/image_raw')

    def image_callback(self, msg):
        """画像トピックを受け取った時に実行されるメインロジック"""
        try:
            # ROSメッセージをOpenCV形式に変換
            color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 1. YOLO推論実行
            results = self.vision.model(color_frame, conf=self.vision.conf_threshold, device='cuda')[0]
            
            people = []
            chairs = []
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                b = box.xyxy[0].cpu().numpy().astype(int)
                
                if cls == 0:    # Person
                    people.append({'bbox': b, 'conf': conf})
                elif cls == 56: # Chair
                    chairs.append({'bbox': b, 'conf': conf})

            # 2. ゲスト接近判定 (画面高さの70%基準)
            status = "searching"
            img_h = color_frame.shape[0]
            for p in people:
                box = p['bbox']
                if (box[3] - box[1]) > img_h * 0.9:
                    status = "guest_arrived"

            # 3. 空席判定 (IoU)
            empty_seats = []
            for c in chairs:
                occupied = False
                for p in people:
                    # vision.pyの重なり判定関数を使用
                    if self.vision._compute_iou(c['bbox'], p['bbox']) > 0.1:
                        occupied = True
                        break
                if not occupied:
                    empty_seats.append(c['bbox'].tolist())

            # 4. JSON結果のパブリッシュ
            res = {
                "status": status,
                "people_count": len(people),
                "empty_seats": empty_seats
            }
            self.detection_pub.publish(String(data=json.dumps(res)))

            # 5. デバッグ表示
            self._show_debug(color_frame, people, chairs, empty_seats)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

    def _show_debug(self, frame, people, chairs, empty_seats):
        for p in people:
            cv2.rectangle(frame, (p['bbox'][0], p['bbox'][1]), (p['bbox'][2], p['bbox'][3]), (255, 0, 0), 2)
        for s in empty_seats:
            cv2.rectangle(frame, (s[0], s[1]), (s[2], s[3]), (0, 255, 0), 3)
            cv2.putText(frame, "EMPTY", (s[0], s[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Receptionist Vision", frame)
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