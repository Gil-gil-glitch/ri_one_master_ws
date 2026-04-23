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

        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.25)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        conf_thresh = self.get_parameter('conf_threshold').get_parameter_value().double_value

        # Main YOLO model (detection)
        from ultralytics import YOLO
        self.yolo = YOLO(model_path)
        # YOLO-World for zero-shot detection
        self.yolo_world = YOLO('yolov8s-worldv2.pt')
        self.yolo_world.set_classes(["paper bag", "chair", "person"])
        # YOLO-Pose for keypoints
        self.yolo_pose = YOLO('yolov8n-pose.pt')

        # CLIP Attribute Detector
        from receptionist_system.core.clip_attributes import ClipAttributeDetector
        self.clip = ClipAttributeDetector()

        # Identity Recognizer (InsightFace)
        from receptionist_system.core.identity import IdentityRecognizer
        try:
            self.identity = IdentityRecognizer(ctx_id=0)
        except Exception as e:
            self.identity = None
            self.get_logger().warn(f"InsightFace not available: {e}")

        # Publisher
        self.detection_pub = self.create_publisher(String, '/receptionist/detections', 10)

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info('Vision Node started: Subscribing to /camera/color/image_raw')


    def image_callback(self, msg):
        """Unified perception pipeline: YOLO, YOLO-World, YOLO-Pose, CLIP, InsightFace"""
        try:
            color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_h, img_w = color_frame.shape[:2]

            # 1. YOLO detection (person/chair)
            results = self.yolo(color_frame, conf=0.25, verbose=False)[0]
            people = []
            chairs = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                b = box.xyxy[0].cpu().numpy().astype(int)
                if cls == 0:
                    people.append({'bbox': b, 'conf': conf})
                elif cls == 56:
                    chairs.append({'bbox': b, 'conf': conf})

            # 2. YOLO-World zero-shot detection
            world_results = self.yolo_world(color_frame, conf=0.25, verbose=False)
            world_objs = []
            for box in world_results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                b = box.xyxy[0].cpu().numpy().astype(int)
                label = self.yolo_world.model.names[cls] if hasattr(self.yolo_world.model, 'names') else str(cls)
                world_objs.append({'bbox': b.tolist(), 'conf': conf, 'label': label})

            # 3. YOLO-Pose for keypoints & pointing
            pose_results = self.yolo_pose(color_frame, verbose=False)
            keypoints = []
            pointing = []
            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                kps = pose_results[0].keypoints
                for person_idx in range(kps.xy.shape[0]):
                    xy = kps.xy[person_idx].cpu().numpy()  # (17, 2)
                    confs = kps.conf[person_idx].cpu().numpy()
                    # Save keypoints for CLIP
                    kp_dict = {}
                    if len(xy) > 6:
                        kp_dict = {
                            'left_eye': xy[1]/[img_w, img_h],
                            'right_eye': xy[2]/[img_w, img_h],
                            'left_ear': xy[3]/[img_w, img_h],
                            'right_ear': xy[4]/[img_w, img_h],
                            'left_shoulder': xy[5]/[img_w, img_h],
                            'right_shoulder': xy[6]/[img_w, img_h],
                        }
                    keypoints.append(kp_dict)
                    # Pointing detection
                    # Right arm: shoulder(6) -> wrist(10)
                    # Left arm: shoulder(5) -> wrist(9)
                    r_conf = min(confs[6], confs[10]) if len(confs) > 10 else 0
                    l_conf = min(confs[5], confs[9]) if len(confs) > 9 else 0
                    if r_conf > l_conf and r_conf > 0.5:
                        shoulder, wrist, arm_label = xy[6], xy[10], "R"
                    elif l_conf > 0.5:
                        shoulder, wrist, arm_label = xy[5], xy[9], "L"
                    else:
                        continue
                    dx = wrist[0] - shoulder[0]
                    dy = wrist[1] - shoulder[1]
                    mag = np.sqrt(dx**2 + dy**2) + 1e-6
                    pointing.append({
                        'person_idx': person_idx,
                        'shoulder': shoulder.tolist(),
                        'wrist': wrist.tolist(),
                        'direction': [float(dx/mag), float(dy/mag)],
                        'arm': arm_label
                    })

            # 4. CLIP attributes & Identity for each person
            people_out = []
            for idx, p in enumerate(people):
                bbox = p['bbox']
                # Find matching keypoints
                lm = keypoints[idx] if idx < len(keypoints) else None
                # CLIP attributes
                attrs = self.clip.detect_attributes(color_frame, tuple(bbox), landmarks=lm, include_debug=False)
                # Identity
                identity = None
                if self.identity is not None:
                    try:
                        name, sim, unc, age, gender = self.identity.get_identity(color_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                        identity = {'name': name, 'similarity': sim, 'uncertainty': unc, 'age': age, 'gender': gender}
                    except Exception as e:
                        identity = {'error': str(e)}
                # Compose output
                people_out.append({
                    'bbox': bbox.tolist(),
                    'conf': float(p['conf']),
                    'attributes': attrs,
                    'identity': identity
                })

            # 5. Guest arrival logic (legacy)
            status = "searching"
            for p in people:
                box = p['bbox']
                if (box[3] - box[1]) > img_h * 0.9:
                    status = "guest_arrived"

            # 6. Empty seat logic (legacy)
            empty_seats = []
            for c in chairs:
                occupied = False
                for p in people:
                    # Simple IoU
                    def iou(boxA, boxB):
                        xA = max(boxA[0], boxB[0])
                        yA = max(boxA[1], boxB[1])
                        xB = min(boxA[2], boxB[2])
                        yB = min(boxA[3], boxB[3])
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                        return interArea / float(boxAArea + boxBArea - interArea)
                    if iou(c['bbox'], p['bbox']) > 0.1:
                        occupied = True
                        break
                if not occupied:
                    empty_seats.append(c['bbox'].tolist())

            # 7. Publish all results
            res = {
                "status": status,
                "people": people_out,
                "chairs": [c['bbox'].tolist() for c in chairs],
                "empty_seats": empty_seats,
                "objects": world_objs,
                "pointing": pointing
            }
            self.detection_pub.publish(String(data=json.dumps(res)))

            # Debug visualization
            self._show_debug(color_frame, people_out, chairs, empty_seats, pointing, world_objs)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

    def _show_debug(self, frame, people, chairs, empty_seats, pointing, world_objs):
        # Draw people
        for idx, p in enumerate(people):
            bbox = p['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # Draw attributes
            attrs = p.get('attributes', {})
            y0 = bbox[1] + 20
            for i, (attr, val) in enumerate(attrs.items()):
                if attr == '_debug': continue
                label = f"{attr}: {val}"
                cv2.putText(frame, label, (bbox[0]+5, y0 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            # Draw identity
            ident = p.get('identity', {})
            if ident and isinstance(ident, dict):
                id_label = f"ID: {ident.get('name','?')} ({ident.get('similarity',0):.2f})"
                cv2.putText(frame, id_label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # Draw chairs
        for s in empty_seats:
            cv2.rectangle(frame, (s[0], s[1]), (s[2], s[3]), (0, 255, 0), 3)
            cv2.putText(frame, "EMPTY", (s[0], s[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Draw objects
        for obj in world_objs:
            b = obj['bbox']
            label = obj.get('label', '?')
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
            cv2.putText(frame, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        # Draw pointing
        for pt in pointing:
            s = pt['shoulder']
            w = pt['wrist']
            cv2.line(frame, (int(s[0]), int(s[1])), (int(w[0]), int(w[1])), (0,255,255), 3)
            dx, dy = pt['direction']
            ext_x = int(w[0] + dx * 80)
            ext_y = int(w[1] + dy * 80)
            cv2.line(frame, (int(w[0]), int(w[1])), (ext_x, ext_y), (0,200,200), 2, cv2.LINE_AA)
            cv2.circle(frame, (int(w[0]), int(w[1])), 8, (0,0,255), -1)
            cv2.circle(frame, (int(s[0]), int(s[1])), 6, (255,0,0), -1)
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