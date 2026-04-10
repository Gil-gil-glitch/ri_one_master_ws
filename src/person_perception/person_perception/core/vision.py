"""
Vision Module: YOLOv8 Wrapper for Person Detection
===================================================
Refactored from project_01's detection.py with class-based architecture.
Provides person detection with depth information using YOLOv8 and Intel RealSense.
"""

import os
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs


class VisionProcessor:
    """
    YOLOv8-based vision processor with RealSense integration.
    Specifically optimized for person detection (COCO class 0).
    """
    
    # COCO class ID for 'person'
    PERSON_CLASS_ID = 0
    
    # COCO class IDs for accessories
    ACCESSORY_CLASS_IDS = {
        24: 'Backpack',
        26: 'Handbag',
        28: 'Suitcase',
        27: 'Tie',
        39: 'Bottle'
    }
    
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        conf_threshold: float = 0.25,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30
    ):
        """
        Initialize the vision processor with YOLOv8 model and RealSense camera.
        
        Args:
            model_path: Path to YOLOv8 model weights
            conf_threshold: Confidence threshold for detections (0-1)
            resolution: Camera resolution (width, height)
            fps: Camera frame rate
        """
        # Resolve model path
        if not os.path.isabs(model_path):
            model_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'config', model_path
            )
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.width, self.height = resolution
        
        # RealSense pipeline setup
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, fps
        )
        self.rs_config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, fps
        )
        
        # Start pipeline and get depth scale
        self.pipeline_profile = self.pipeline.start(self.rs_config)
        depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Color detection utilities
        self._color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "brown": (165, 42, 42)
        }
    
    def __del__(self):
        """Ensure proper cleanup of RealSense pipeline."""
        try:
            self.pipeline.stop()
        except Exception:
            pass
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get aligned color and depth frames from RealSense camera.
        
        Returns:
            Tuple of (color_image, depth_image) or (None, None) if frames invalid
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image
    
    def detect_persons(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Detect persons in the image with optional depth information.
        
        Args:
            color_image: BGR image to process
            depth_image: Optional depth image for distance calculation
            
        Returns:
            List of person detection dictionaries with:
            - bbox: (x1, y1, x2, y2)
            - confidence: detection confidence
            - center: (cx, cy)
            - distance_m: distance in meters (if depth provided)
            - area: bounding box area (for sorting)
            - dominant_color: detected dominant color
        """
        results = self.model(color_image, conf=self.conf_threshold, verbose=False)
        persons = []
        
        for det in results[0].boxes:
            cls = int(det.cls[0])
            
            # Filter for person class only
            if cls != self.PERSON_CLASS_ID:
                continue
            
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            
            # Calculate center and area
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            
            # Get distance if depth is available
            distance_m = None
            if depth_image is not None:
                # Clamp to image bounds
                cx_clamped = max(0, min(cx, self.width - 1))
                cy_clamped = max(0, min(cy, self.height - 1))
                
                # Get median depth in small region for robustness
                roi_size = 5
                y_start = max(0, cy_clamped - roi_size)
                y_end = min(self.height, cy_clamped + roi_size + 1)
                x_start = max(0, cx_clamped - roi_size)
                x_end = min(self.width, cx_clamped + roi_size + 1)
                
                depth_roi = depth_image[y_start:y_end, x_start:x_end]
                if depth_roi.size > 0:
                    distance_m = float(np.median(depth_roi)) * self.depth_scale
            
            # Get dominant color from bounding box
            dominant_color = self._get_dominant_color(
                color_image[max(0, y1):min(self.height, y2),
                           max(0, x1):min(self.width, x2)]
            )
            
            persons.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'center': (cx, cy),
                'distance_m': distance_m,
                'area': area,
                'dominant_color': dominant_color
            })
        
        return persons
    
    def get_closest_person(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        Get the closest (largest bounding box) person in the image.
        
        The largest bounding box typically corresponds to the closest person
        in a depth-agnostic manner.
        
        Args:
            color_image: BGR image to process
            depth_image: Optional depth image for distance calculation
            
        Returns:
            Person detection dictionary or None if no person detected
        """
        persons = self.detect_persons(color_image, depth_image)
        
        if not persons:
            return None
        
        # Sort by area (descending) and return the largest
        return max(persons, key=lambda p: p['area'])
    
    def _get_dominant_color(self, crop: np.ndarray, k: int = 1) -> str:
        """
        Find the dominant color in an image crop using k-means clustering.
        
        Args:
            crop: Image crop (BGR format)
            k: Number of clusters for k-means
            
        Returns:
            Name of the dominant color
        """
        if crop is None or crop.size == 0:
            return "unknown"
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100, 0.2
        )
        _, _, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        dominant_rgb = centers[0].astype(int)
        return self._rgb_to_color_name(dominant_rgb)
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Map RGB color to closest predefined color name."""
        r, g, b = rgb
        min_dist = float('inf')
        closest = "unknown"
        
        for name, (cr, cg, cb) in self._color_map.items():
            dist = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
            if dist < min_dist:
                min_dist = dist
                closest = name
        
        return closest
    
    def draw_detection(
        self,
        image: np.ndarray,
        person: Dict,
        label: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw detection box and label on image.
        
        Args:
            image: Image to annotate
            person: Person detection dictionary
            label: Optional custom label (defaults to confidence + distance)
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        x1, y1, x2, y2 = person['bbox']
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label
        if label is None:
            label = f"Person {person['confidence']:.2f}"
            if person['distance_m'] is not None:
                label += f" | {person['distance_m']:.2f}m"
        
        # Draw label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            annotated,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            (0, 255, 0), -1
        )
        cv2.putText(
            annotated, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )
        
        return annotated
    
    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First box (x1, y1, x2, y2)
            box2: Second box (x1, y1, x2, y2)
            
        Returns:
            IoU score (0-1)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        inter_width = max(0, x2 - x1)
        inter_height = max(0, y2 - y1)
        inter_area = inter_width * inter_height
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _classify_shirt_color_hsv(self, torso_crop: np.ndarray) -> str:
        """
        Classify shirt color using HSV color space analysis.
        
        Args:
            torso_crop: BGR image of torso region
            
        Returns:
            Color label: Red, Blue, Green, Black, White, or Grey
        """
        if torso_crop is None or torso_crop.size == 0:
            return "Unknown"
        
        # Convert to HSV
        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        
        # Get average/dominant HSV values
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        # Classify based on saturation and value first
        # Low saturation indicates grayscale (Black/White/Grey)
        if s_mean < 40:
            if v_mean < 50:
                return "Black"
            elif v_mean > 200:
                return "White"
            else:
                return "Grey"
        
        # High saturation - classify by hue
        # Hue ranges (OpenCV uses 0-179 for hue)
        # Red: 0-10 or 160-179
        # Orange: 10-25
        # Yellow: 25-35
        # Green: 35-85
        # Blue: 85-130
        # Purple/Magenta: 130-160
        
        if h_mean < 10 or h_mean > 160:
            return "Red"
        elif h_mean < 35:
            return "Red"  # Orange/Yellow classified as Red for simplicity
        elif h_mean < 85:
            return "Green"
        elif h_mean < 130:
            return "Blue"
        else:
            return "Red"  # Purple/Magenta -> Red for simplicity
    
    def detect_attributes(
        self,
        image: np.ndarray,
        person_box: Tuple[int, int, int, int]
    ) -> List[str]:
        """
        Detect clothing and accessory attributes for a person.
        
        Args:
            image: Full frame (BGR image)
            person_box: XYXY coordinates of the person bounding box
            
        Returns:
            List of attribute strings (e.g., ["Red_Shirt", "Backpack"])
        """
        attributes = []
        
        # Logic A: Accessory detection via YOLO
        # Run YOLO on full image to detect accessories
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        for det in results[0].boxes:
            cls_id = int(det.cls[0])
            
            # Check if this is an accessory class
            if cls_id in self.ACCESSORY_CLASS_IDS:
                acc_box = tuple(map(int, det.xyxy[0]))
                
                # Check IoU overlap with person box
                iou = self._compute_iou(person_box, acc_box)
                if iou > 0.1:
                    accessory_name = self.ACCESSORY_CLASS_IDS[cls_id]
                    if accessory_name not in attributes:
                        attributes.append(accessory_name)
        
        # Logic B: Shirt color via HSV
        # Crop the "Torso" area (center-middle 30% of person_box)
        x1, y1, x2, y2 = person_box
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Torso region: horizontally centered, vertically in upper-middle area
        # Center 50% width, 20-50% height (roughly upper torso area)
        torso_x1 = x1 + int(box_width * 0.25)
        torso_x2 = x1 + int(box_width * 0.75)
        torso_y1 = y1 + int(box_height * 0.20)
        torso_y2 = y1 + int(box_height * 0.50)
        
        # Clamp to image bounds
        h, w = image.shape[:2]
        torso_x1 = max(0, min(torso_x1, w - 1))
        torso_x2 = max(0, min(torso_x2, w))
        torso_y1 = max(0, min(torso_y1, h - 1))
        torso_y2 = max(0, min(torso_y2, h))
        
        if torso_x2 > torso_x1 and torso_y2 > torso_y1:
            torso_crop = image[torso_y1:torso_y2, torso_x1:torso_x2]
            shirt_color = self._classify_shirt_color_hsv(torso_crop)
            if shirt_color != "Unknown":
                attributes.insert(0, f"{shirt_color}_Shirt")
        
        return attributes

