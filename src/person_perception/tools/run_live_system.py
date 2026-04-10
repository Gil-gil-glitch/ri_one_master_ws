#!/usr/bin/env python3
"""
Live System Simulator: Pure Python Webcam-Based Perception
===========================================================
Runs the full Attribute/Biometric detection pipeline using a webcam.
This script replaces 'ros2 run' for development without ROS 2.

Outputs the NLP-compliant JSON to console and displays annotated video.
"""

import json
import sys
import cv2
import numpy as np

# Add parent path for imports
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/tools/', 1)[0])

import time

from person_perception.core.vision import VisionProcessor
from person_perception.core.identity import IdentityRecognizer
from person_perception.core.clip_attributes import ClipAttributeDetector


class LivePerceptionSystem:
    """
    Standalone perception system using webcam instead of RealSense.
    Produces the exact JSON format required by the NLP team.
    """
    
    # Active Perception thresholds
    UNCERTAINTY_THRESHOLD = 0.4
    SIMILARITY_THRESHOLD = 0.65
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize the live perception system.
        
        Args:
            camera_id: Webcam device ID (default 0)
        """
        print("Initializing Live Perception System...")
        print("=" * 60)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize YOLO for person detection
        print("Loading YOLO models...")
        self.vision = VisionProcessorWebcam()
        
        # Initialize InsightFace for identity + biometrics
        print("Loading InsightFace model (GPU)...")
        try:
            self.identity = IdentityRecognizer(ctx_id=0)  # GPU
            print(f"Known identities: {self.identity.get_known_identities()}")
        except Exception as e:
            print(f"InsightFace init failed: {e}")
            self.identity = None

        # Initialize CLIP for attributes
        print("Loading CLIP (ViT-B/32)...")
        try:
            self.clip = ClipAttributeDetector()
            self.use_clip = True
        except Exception as e:
            print(f"Warning: CLIP init failed ({e}). Attributes disabled.")
            self.use_clip = False
            
        # State for CLIP throttling
        self.last_attributes = []
        self.last_clip_time = 0
        self.clip_interval = 1.0  # 1Hz
        
        print("=" * 60)
        print("System ready! Press 'q' to quit, 's' to save screenshot")
        print("=" * 60)
    
    def run(self):
        """Main processing loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame and get perception data
            perception_data = self._process_frame(frame)
            
            # Print JSON to console (one line per frame)
            print(json.dumps(perception_data))
            
            # Draw visualization
            annotated = self._draw_visualization(frame, perception_data)
            
            # Show frame
            cv2.imshow("Person Perception Live System", annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("perception_screenshot.png", annotated)
                print("Screenshot saved!")
        
        self.cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and return NLP-compliant JSON.
        
        Returns:
            {
                "is_human": true,
                "id": "Jonathan",
                "uncertainty": 0.12,
                "biometrics": {
                    "estimated_age": 22,
                    "gender": "M"
                },
                "attributes": ["Black_Shirt", "Glasses", "Backpack"],
                "action": "GREET"
            }
        """
        # Detect person using YOLO
        person = self.vision.get_closest_person(frame)
        
        if person is None:
            return {
                "is_human": False,
                "id": None,
                "uncertainty": 1.0,
                "biometrics": {
                    "estimated_age": None,
                    "gender": None
                },
                "attributes": [],
                "action": "OBSERVE"
            }
        
        # Get identity + biometrics
        identity_name = "Unknown"
        similarity_score = 0.0
        uncertainty_score = 1.0
        age = None
        gender = None
        
        if self.identity is not None:
            try:
                # Updated get_identity returns 5 values
                result = self.identity.get_identity(frame)
                if len(result) == 5:
                    identity_name, similarity_score, uncertainty_score, age, gender = result
                else:
                    identity_name, similarity_score, uncertainty_score = result
                    age, gender = None, None
            except Exception as e:
                print(f"Identity error: {e}")
        
        if self.identity is not None:
            try:
                # Updated get_identity returns 5 values
                result = self.identity.get_identity(frame)
                if len(result) == 5:
                    identity_name, similarity_score, uncertainty_score, age, gender = result
                else:
                    identity_name, similarity_score, uncertainty_score = result
                    age, gender = None, None
            except Exception as e:
                print(f"Identity error: {e}")
        
        # Determine action using state machine
        action = self._determine_action(uncertainty_score, similarity_score)

        # Get clothing/accessories via CLIP (1Hz Throttle)
        # We use the person['bbox'] from YOLO to crop the person for CLIP
        # This fixes the "partial body" issue by focusing CLIP on the detected person
        current_time = time.time()
        if self.use_clip and action != "OBSERVE":
            if current_time - self.last_clip_time > self.clip_interval:
                self.last_clip_time = current_time
                try:
                    if person:
                        self.last_attributes = self.clip.detect_attributes(frame, person['bbox'])
                except Exception as e:
                    print(f"CLIP error: {e}")
        
        attributes = self.last_attributes

        # Convert gender to single char format
        gender_char = None
        if gender == "Male":
            gender_char = "M"
        elif gender == "Female":
            gender_char = "F"
        
        # Store for visualization
        self._last_person = person
        self._last_identity = identity_name
        self._last_action = action
        self._last_attributes = attributes
        self._last_age = age
        self._last_gender = gender
        
        return {
            "is_human": True,
            "id": identity_name if identity_name != "Unknown" else "Unknown",
            "uncertainty": round(float(uncertainty_score), 3),
            "biometrics": {
                "estimated_age": int(age) if age is not None else None,
                "gender": gender_char
            },
            "attributes": attributes,
            "action": action
        }
    
    def _determine_action(self, uncertainty: float, similarity: float) -> str:
        """Determine action based on Active Perception state machine."""
        if uncertainty > self.UNCERTAINTY_THRESHOLD:
            return "ASK_CLARIFICATION"
        elif similarity > self.SIMILARITY_THRESHOLD:
            return "GREET"
        else:
            return "LEARN"
    
    def _draw_visualization(self, frame: np.ndarray, data: dict) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        annotated = frame.copy()
        
        # Draw action banner at top
        action = data.get('action', 'OBSERVE')
        action_colors = {
            'GREET': (0, 255, 0),          # Green
            'ASK_CLARIFICATION': (0, 255, 255),  # Yellow
            'LEARN': (0, 165, 255),        # Orange
            'OBSERVE': (128, 128, 128)     # Gray
        }
        action_color = action_colors.get(action, (255, 255, 255))
        
        cv2.putText(
            annotated, f"Action: {action}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2
        )
        
        # Draw ID and biometrics
        identity = data.get('id', 'Unknown') or 'Unknown'
        biometrics = data.get('biometrics', {})
        age = biometrics.get('estimated_age')
        gender = biometrics.get('gender')
        
        bio_str = ""
        if gender:
            bio_str += gender
        if age:
            bio_str += f", {age}yo" if bio_str else f"{age}yo"
        
        cv2.putText(
            annotated, f"ID: {identity} [{bio_str}]" if bio_str else f"ID: {identity}", 
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        # Draw attributes
        attributes = data.get('attributes', [])
        cv2.putText(
            annotated, f"Attrs: {attributes}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        # Draw person bounding box if available
        if hasattr(self, '_last_person') and self._last_person:
            x1, y1, x2, y2 = self._last_person['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), action_color, 2)
            
            # Draw label on box with name, age, action
            label_parts = [identity]
            if bio_str:
                label_parts.append(f"({bio_str})")
            label_parts.append(f"| {action}")
            label = " ".join(label_parts)
            
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, action_color, 2
            )
        
        return annotated
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nSystem shutdown complete.")


class VisionProcessorWebcam:
    """
    Webcam-based Vision Processor (no RealSense dependency).
    Uses same YOLO detection logic as the RealSense version.
    """
    
    PERSON_CLASS_ID = 0
    ACCESSORY_CLASS_IDS = {
        24: 'Backpack',
        26: 'Handbag',
        28: 'Suitcase',
        27: 'Tie',
        39: 'Bottle'
    }
    
    def __init__(self, model_path: str = 'yolo26n.pt', conf_threshold: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def detect_persons(self, image: np.ndarray) -> list:
        """Detect all persons in image."""
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        persons = []
        
        for det in results[0].boxes:
            cls = int(det.cls[0])
            if cls != self.PERSON_CLASS_ID:
                continue
            
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            area = (x2 - x1) * (y2 - y1)
            
            persons.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'area': area
            })
        
        return persons
    
    def get_closest_person(self, image: np.ndarray) -> dict:
        """Get largest (closest) person."""
        persons = self.detect_persons(image)
        if not persons:
            return None
        return max(persons, key=lambda p: p['area'])
    
    def detect_attributes(self, image: np.ndarray, person_box: tuple) -> list:
        """Detect clothing and accessory attributes."""
        attributes = []
        
        # Accessory detection via YOLO
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        for det in results[0].boxes:
            cls_id = int(det.cls[0])
            if cls_id in self.ACCESSORY_CLASS_IDS:
                acc_box = tuple(map(int, det.xyxy[0]))
                iou = self._compute_iou(person_box, acc_box)
                if iou > 0.1:
                    name = self.ACCESSORY_CLASS_IDS[cls_id]
                    if name not in attributes:
                        attributes.append(name)
        
        # Shirt color via HSV
        x1, y1, x2, y2 = person_box
        box_width = x2 - x1
        box_height = y2 - y1
        
        torso_x1 = x1 + int(box_width * 0.25)
        torso_x2 = x1 + int(box_width * 0.75)
        torso_y1 = y1 + int(box_height * 0.20)
        torso_y2 = y1 + int(box_height * 0.50)
        
        h, w = image.shape[:2]
        torso_x1 = max(0, min(torso_x1, w - 1))
        torso_x2 = max(0, min(torso_x2, w))
        torso_y1 = max(0, min(torso_y1, h - 1))
        torso_y2 = max(0, min(torso_y2, h))
        
        if torso_x2 > torso_x1 and torso_y2 > torso_y1:
            torso_crop = image[torso_y1:torso_y2, torso_x1:torso_x2]
            shirt_color = self._classify_shirt_color(torso_crop)
            if shirt_color != "Unknown":
                attributes.insert(0, f"{shirt_color}_Shirt")
        
        return attributes
    
    def _compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _classify_shirt_color(self, crop: np.ndarray) -> str:
        """Classify shirt color using HSV."""
        if crop is None or crop.size == 0:
            return "Unknown"
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        if s_mean < 40:
            if v_mean < 50:
                return "Black"
            elif v_mean > 200:
                return "White"
            else:
                return "Grey"
        
        if h_mean < 10 or h_mean > 160:
            return "Red"
        elif h_mean < 35:
            return "Red"
        elif h_mean < 85:
            return "Green"
        elif h_mean < 130:
            return "Blue"
        else:
            return "Red"

class PipelineSimulator:
    """
    Simulates the 3-node ROS2 architecture locally without ROS 2.
    
    Pipeline (same as ROS 2 architecture):
      VisionNode -> PersonTracker -> TaskPlanner
    
    This lets you test the full person learning flow using just a webcam.
    """
    
    def __init__(self, camera_id: int = 0):
        print("Initializing Pipeline Simulator (3-Node Architecture)...")
        print("=" * 60)
        
        # === Node 1: Vision Node components ===
        print("[VisionNode] Loading YOLO models...")
        self.vision = VisionProcessorWebcam()
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # === Node 2: Person Tracker components ===
        print("[PersonTracker] Loading InsightFace (GPU)...")
        try:
            from person_perception.nodes.person_tracker_node import PersonTrackerNode
            self.tracker = PersonTrackerNode()
        except Exception as e:
            print(f"  PersonTracker init failed: {e}")
            self.tracker = None
        
        # === Node 3: Task Planner components ===
        print("[TaskPlanner] Initializing coordinator...")
        try:
            from person_perception.nodes.task_planner_node import TaskPlannerNode
            self.planner = TaskPlannerNode()
        except Exception as e:
            print(f"  TaskPlanner init failed: {e}")
            self.planner = None
        
        # CLIP for attributes (optional, 1Hz)
        print("[VisionNode] Loading CLIP (ViT-B/32)...")
        try:
            self.clip = ClipAttributeDetector()
            self.use_clip = True
        except Exception as e:
            print(f"  CLIP init failed ({e}). Attributes via HSV only.")
            self.use_clip = False
        
        self.last_clip_time = 0
        self.last_attributes = []
        self.clip_interval = 1.0
        
        print("=" * 60)
        print("Pipeline ready! Press 'q' to quit, 's' to save screenshot")
        print("=" * 60)
    
    def run(self):
        """Main pipeline loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # === Stage 1: Vision Node (person detection) ===
            persons = self.vision.detect_persons(frame)
            
            # Get CLIP attributes for closest person (1Hz throttle)
            current_time = time.time()
            if persons and self.use_clip:
                if current_time - self.last_clip_time > self.clip_interval:
                    self.last_clip_time = current_time
                    try:
                        closest = max(persons, key=lambda p: p['area'])
                        self.last_attributes = self.clip.detect_attributes(
                            frame, closest['bbox']
                        )
                    except Exception:
                        pass
            elif not persons:
                self.last_attributes = []
            
            # Build detection message (simulating /person_detection topic)
            detection_msg = {
                'timestamp': int(current_time * 1000),
                'persons': [
                    {
                        'bbox': list(p['bbox']),
                        'confidence': round(p['confidence'], 4),
                        'distance_m': None,
                        'attributes': self.last_attributes if p == max(
                            persons, key=lambda x: x['area']
                        ) else []
                    }
                    for p in persons
                ] if persons else []
            }
            
            # === Stage 2: Person Tracker (identity assignment) ===
            tracked_msg = {'tracked_persons': []}
            if self.tracker is not None and persons:
                self.tracker.set_frame(frame)
                tracked_msg = self.tracker.process_detections(
                    detection_msg['persons'], frame
                )
            
            # === Stage 3: Task Planner (action coordination) ===
            actions = []
            if self.planner is not None and tracked_msg.get('tracked_persons'):
                actions = self.planner.process_tracked_persons(tracked_msg)
            
            # === Visualization ===
            annotated = self._draw_pipeline_viz(
                frame, detection_msg, tracked_msg, actions
            )
            
            cv2.imshow("Person Learning System - Pipeline", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("pipeline_screenshot.png", annotated)
                print("Screenshot saved!")
        
        self.cleanup()
    
    def _draw_pipeline_viz(
        self,
        frame: np.ndarray,
        detections: dict,
        tracked: dict,
        actions: list
    ) -> np.ndarray:
        """Draw the pipeline visualization with tracking info."""
        annotated = frame.copy()
        
        tracked_persons = tracked.get('tracked_persons', [])
        
        # Action colors
        action_colors = {
            'GREET': (0, 255, 0),
            'ASK_CLARIFICATION': (0, 255, 255),
            'LEARN': (0, 165, 255),
            'OBSERVE': (128, 128, 128)
        }
        
        # Header
        cv2.putText(
            annotated, f"Pipeline Mode | Tracking: {len(tracked_persons)} persons",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        # Draw each tracked person
        for tp in tracked_persons:
            bbox = tp.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            action = tp.get('action', 'OBSERVE')
            color = action_colors.get(action, (255, 255, 255))
            name = tp.get('name', 'Unknown')
            track_id = tp.get('track_id', '?')
            similarity = tp.get('similarity', 0.0)
            uncertainty = tp.get('uncertainty', 1.0)
            
            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Action banner
            cv2.putText(
                annotated, f"ACTION: {action}", (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Identity label
            label = f"[{track_id}] {name} ({similarity:.2f})"
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )
            
            # Attributes
            attrs = tp.get('attributes', [])
            if attrs:
                attr_text = ', '.join(attrs[:4])
                cv2.putText(
                    annotated, attr_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
                )
            
            # Biometrics
            bio = tp.get('biometrics', {})
            bio_parts = []
            if bio.get('gender'):
                bio_parts.append(bio['gender'])
            if bio.get('age'):
                bio_parts.append(f"Age:{bio['age']}")
            if bio_parts:
                cv2.putText(
                    annotated, ' | '.join(bio_parts), (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1
                )
        
        # Print actions to console
        for action in actions:
            action_type = action.get('action_type', '?')
            target = action.get('target_name', 'Unknown')
            dialogue = action.get('dialogue', '')
            print(f"  [ACTION] {action_type} -> {target}: \"{dialogue}\"")
        
        return annotated
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        if self.planner is not None:
            self.planner.shutdown()
        print("\nPipeline shutdown complete.")


def main():
    """Entry point for the live perception system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Person Perception Live System'
    )
    parser.add_argument(
        '--mode', choices=['legacy', 'pipeline'], default='pipeline',
        help='legacy = monolithic node, pipeline = 3-node architecture (default)'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device ID (default: 0)'
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    if args.mode == 'pipeline':
        print("  PERSON LEARNING SYSTEM — Pipeline Mode (3-Node)")
        print("  VisionNode -> PersonTracker -> TaskPlanner")
    else:
        print("  PERSON PERCEPTION LIVE SYSTEM — Legacy Mode")
        print("  Pure Python Webcam-Based Biometric & Attribute Detection")
    print("=" * 60 + "\n")
    
    try:
        if args.mode == 'pipeline':
            system = PipelineSimulator(camera_id=args.camera)
        else:
            system = LivePerceptionSystem(camera_id=args.camera)
        system.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

