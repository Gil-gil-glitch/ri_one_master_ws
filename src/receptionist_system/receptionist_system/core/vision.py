import cv2
import numpy as np
from ultralytics import YOLO

class VisionProcessor:
    PERSON_CLASS_ID = 0
    CHAIR_CLASS_ID = 56  # COCOの椅子IDを追加

    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25, resolution: tuple = (640, 480), fps: int = 30):
        # GPU (CUDA) を使用するように設定
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def get_frames(self):
        """
        RealSenseからカラーフレームと深度フレームを取得する
        """
        try:
            # フレームセットを待機 (30FPSなのでタイムアウトを設定)
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # 深度フレームをカラーに位置合わせ
            if hasattr(self, 'align'):
                frames = self.align.process(frames)
                
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            # numpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            return color_image, depth_image
        except Exception as e:
            print(f"Error fetching frames: {e}")
            return None

    def detect_assets(self, frame):
        """人と椅子を同時に検知し、空席を判定する"""
        results = self.model(frame, conf=self.conf_threshold, device='cuda')[0]
        
        people = []
        chairs = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            b = box.xyxy[0].cpu().numpy().astype(int)
            
            if cls == self.PERSON_CLASS_ID:
                people.append({'bbox': b, 'conf': conf})
            elif cls == self.CHAIR_CLASS_ID:
                chairs.append({'bbox': b, 'conf': conf})

        # 空席判定ロジック
        empty_seats = []
        for c in chairs:
            is_occupied = False
            for p in people:
                if self._compute_iou(c['bbox'], p['bbox']) > 0.1: # 10%重なれば使用中
                    is_occupied = True
                    break
            if not is_occupied:
                empty_seats.append(c['bbox'])
                
        return people, empty_seats

    def _compute_iou(self, boxA, boxB):
        """リファレンスにある重なり判定ロジック"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)