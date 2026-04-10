"""
Person Tracker Node: Identity Assignment & Cross-Frame Tracking
=================================================================
Dedicated ROS 2 node for assigning persistent IDs to detected persons.
Subscribes to /person_detection, publishes tracked persons to /tracked_person.

Part of the 3-node Person Learning System architecture:
  /vision_node -> /person_tracker -> /task_planner

Tracking strategy:
  1. Receive person detections (bboxes) from /vision_node
  2. For each detection, extract face embedding via InsightFace
  3. Match against known identities using cosine similarity
  4. Maintain cross-frame tracking using IoU + embedding association
  5. Apply Active Perception state machine (GREET/LEARN/ASK/OBSERVE)
  6. Publish tracked persons with persistent IDs to /tracked_person
"""

import json
import time
import base64
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

# Conditional import for ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object
    String = object  # Placeholder for type annotations

# Conditional import for InsightFace
try:
    from ..core.identity import IdentityRecognizer
    IDENTITY_AVAILABLE = True
except ImportError:
    IDENTITY_AVAILABLE = False


class TrackedPerson:
    """Represents a person being tracked across frames."""
    
    _next_id = 1
    
    def __init__(self, bbox: List[int], embedding: Optional[np.ndarray] = None):
        self.track_id = TrackedPerson._next_id
        TrackedPerson._next_id += 1
        
        self.bbox = bbox
        self.embedding = embedding
        self.name = 'Unknown'
        self.similarity = 0.0
        self.uncertainty = 1.0
        self.age = None
        self.gender = None
        self.attributes = []
        self.frames_seen = 1
        self.frames_missing = 0
        self.last_seen = time.time()
    
    def update(
        self,
        bbox: List[int],
        embedding: Optional[np.ndarray] = None,
        name: str = 'Unknown',
        similarity: float = 0.0,
        uncertainty: float = 1.0,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        attributes: Optional[List[str]] = None
    ):
        """Update track with new detection data."""
        self.bbox = bbox
        if embedding is not None:
            self.embedding = embedding
        self.name = name
        self.similarity = similarity
        self.uncertainty = uncertainty
        if age is not None:
            self.age = age
        if gender is not None:
            self.gender = gender
        if attributes is not None:
            self.attributes = attributes
        self.frames_seen += 1
        self.frames_missing = 0
        self.last_seen = time.time()


class PersonTrackerNode(Node):
    """
    ROS 2 Node for person identity tracking.
    
    Responsibility:
    - Assign persistent IDs to detected persons
    - Maintain tracking across frames using IoU + face embeddings
    - Apply Active Perception state machine
    - Publish tracked person data to /tracked_person
    
    Subscribes: /person_detection (from /vision_node)
    Publishes:  /tracked_person
    """
    
    # Active Perception thresholds
    UNCERTAINTY_THRESHOLD = 0.4
    SIMILARITY_THRESHOLD = 0.65
    
    # Tracking parameters
    IOU_THRESHOLD = 0.3          # Minimum IoU for spatial matching
    MAX_MISSING_FRAMES = 15      # Remove track after N frames without detection
    EMBEDDING_MATCH_THRESHOLD = 0.5  # Minimum similarity for re-identification
    
    def __init__(self):
        if ROS2_AVAILABLE:
            super().__init__('person_tracker')
            
            self.declare_parameter('show_debug_window', True)
            self.declare_parameter('embeddings_path', '')
            
            self.show_debug = self.get_parameter('show_debug_window').value
            embeddings_path = self.get_parameter('embeddings_path').value
        else:
            self.show_debug = True
            embeddings_path = ''
        
        # Active tracks
        self.tracks: List[TrackedPerson] = []
        
        # Initialize identity recognizer
        self.identity: Optional[IdentityRecognizer] = None
        if IDENTITY_AVAILABLE:
            try:
                self._log('Initializing IdentityRecognizer (InsightFace GPU)...')
                self.identity = IdentityRecognizer(ctx_id=0)
                
                if embeddings_path:
                    import os
                    if os.path.exists(embeddings_path):
                        self.identity.load_embeddings(embeddings_path)
                
                self._log(
                    f'Known identities: {self.identity.get_known_identities()}'
                )
            except Exception as e:
                self._log(f'InsightFace init failed: {e}', level='warn')
                self.identity = None
        else:
            self._log(
                'InsightFace not available. Face recognition disabled.',
                level='warn'
            )
        
        # Current frame (needed for face extraction — set by process or externally)
        self._current_frame: Optional[np.ndarray] = None
        
        # Publisher (ROS 2 only)
        self.publisher = None
        if ROS2_AVAILABLE:
            self.publisher = self.create_publisher(
                String,
                '/tracked_person',
                10
            )
            
            # Subscriber
            self.subscription = self.create_subscription(
                String,
                '/person_detection',
                self._detection_callback,
                10
            )
        
        # Debug window
        if self.show_debug:
            self.window_name = "Person Tracker - Identity"
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 960, 720)
            except Exception as e:
                self._log(f'OpenCV window init failed: {e}', level='warn')
                self.show_debug = False
        
        self._log('Person Tracker Node started!')
    
    def _log(self, msg: str, level: str = 'info'):
        """Log via ROS 2 or print."""
        if ROS2_AVAILABLE and hasattr(self, 'get_logger'):
            logger = self.get_logger()
            if level == 'error':
                logger.error(msg)
            elif level == 'warn':
                logger.warn(msg)
            else:
                logger.info(msg)
        else:
            print(f'[PersonTracker] [{level.upper()}] {msg}')
    
    def _detection_callback(self, msg: String):
        """ROS 2 callback for /person_detection messages."""
        try:
            data = json.loads(msg.data)
            
            # Decode image if available
            image_base64 = data.get('image_base64', "")
            if image_base64:
                try:
                    img_data = base64.b64decode(image_base64)
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    self.set_frame(frame)
                except Exception as e:
                    self._log(f'Image decoding failed: {e}', level='warn')
            
            # Process detections
            self.process_detections(data.get('persons', []))
        except json.JSONDecodeError as e:
            self._log(f'Invalid JSON: {e}', level='warn')
    
    def set_frame(self, frame: np.ndarray):
        """Set the current frame for face extraction (used in standalone mode)."""
        self._current_frame = frame
    
    def process_detections(
        self,
        detections: List[Dict],
        frame: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Process person detections: assign IDs, track, identify.
        
        Args:
            detections: List of person dicts with 'bbox', 'confidence', etc.
            frame: Optional BGR image for face recognition
            
        Returns:
            Message dict ready for publishing to /tracked_person
        """
        if frame is not None:
            self._current_frame = frame
        
        # Step 1: Match detections to existing tracks using IoU
        matched, unmatched_dets, unmatched_tracks = self._match_detections(
            detections
        )
        
        # Step 2: Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            
            # Extract face identity if frame is available
            identity_info = self._get_identity(det['bbox'])
            
            track.update(
                bbox=det['bbox'],
                embedding=identity_info.get('embedding'),
                name=identity_info.get('name', 'Unknown'),
                similarity=identity_info.get('similarity', 0.0),
                uncertainty=identity_info.get('uncertainty', 1.0),
                age=identity_info.get('age'),
                gender=identity_info.get('gender'),
                attributes=det.get('attributes', [])
            )
        
        # Step 3: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            identity_info = self._get_identity(det['bbox'])
            
            new_track = TrackedPerson(
                bbox=det['bbox'],
                embedding=identity_info.get('embedding')
            )
            new_track.name = identity_info.get('name', 'Unknown')
            new_track.similarity = identity_info.get('similarity', 0.0)
            new_track.uncertainty = identity_info.get('uncertainty', 1.0)
            new_track.age = identity_info.get('age')
            new_track.gender = identity_info.get('gender')
            new_track.attributes = det.get('attributes', [])
            
            self.tracks.append(new_track)
        
        # Step 4: Handle unmatched tracks (increment missing counter)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].frames_missing += 1
        
        # Step 5: Remove stale tracks
        self.tracks = [
            t for t in self.tracks
            if t.frames_missing <= self.MAX_MISSING_FRAMES
        ]
        
        # Step 6: Build and publish message
        msg_data = self._build_message()
        
        if self.publisher is not None:
            msg = String()
            msg.data = json.dumps(msg_data)
            self.publisher.publish(msg)
        
        # Debug visualization
        if self.show_debug and self._current_frame is not None:
            self._visualize(self._current_frame)
        
        return msg_data
    
    def _get_identity(self, bbox: List[int]) -> Dict:
        """Extract face identity from current frame at given bbox."""
        result = {
            'name': 'Unknown',
            'similarity': 0.0,
            'uncertainty': 1.0,
            'age': None,
            'gender': None,
            'embedding': None
        }
        
        if self.identity is None or self._current_frame is None:
            return result
        
        try:
            name, similarity, uncertainty, age, gender = \
                self.identity.get_identity(self._current_frame)
            
            result['name'] = name
            result['similarity'] = similarity
            result['uncertainty'] = uncertainty
            result['age'] = age
            result['gender'] = gender
            
            # Also try to get the embedding for tracking
            faces = self.identity.detect_faces(self._current_frame)
            if faces:
                result['embedding'] = faces[0].get('embedding')
        except Exception as e:
            self._log(f'Identity extraction error: {e}', level='warn')
        
        return result
    
    def _match_detections(
        self,
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match new detections to existing tracks using IoU.
        
        Returns:
            (matched_pairs, unmatched_detection_indices, unmatched_track_indices)
        """
        if not self.tracks or not detections:
            return (
                [],
                list(range(len(detections))),
                list(range(len(self.tracks)))
            )
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._compute_iou(
                    track.bbox, det['bbox']
                )
        
        # Greedy matching (highest IoU first)
        matched = []
        used_tracks = set()
        used_dets = set()
        
        while True:
            # Find max IoU
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(
                np.argmax(iou_matrix), iou_matrix.shape
            )
            max_iou = iou_matrix[max_idx]
            
            if max_iou < self.IOU_THRESHOLD:
                break
            
            t_idx, d_idx = max_idx
            if t_idx not in used_tracks and d_idx not in used_dets:
                matched.append((int(t_idx), int(d_idx)))
                used_tracks.add(t_idx)
                used_dets.add(d_idx)
            
            iou_matrix[t_idx, d_idx] = 0  # Remove this pair
        
        unmatched_dets = [
            i for i in range(len(detections)) if i not in used_dets
        ]
        unmatched_tracks = [
            i for i in range(len(self.tracks)) if i not in used_tracks
        ]
        
        return matched, unmatched_dets, unmatched_tracks
    
    @staticmethod
    def _compute_iou(
        box1: List[int],
        box2: List[int]
    ) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_action(
        self,
        uncertainty: float,
        similarity: float,
        person_detected: bool = True
    ) -> str:
        """
        Active Perception state machine.
        
        Decision Logic:
        1. No person detected -> OBSERVE
        2. uncertainty > 0.4 -> ASK_CLARIFICATION
        3. similarity > 0.65 -> GREET
        4. Else -> LEARN
        """
        if not person_detected:
            return 'OBSERVE'
        if uncertainty > self.UNCERTAINTY_THRESHOLD:
            return 'ASK_CLARIFICATION'
        elif similarity > self.SIMILARITY_THRESHOLD:
            return 'GREET'
        else:
            return 'LEARN'
    
    def _build_message(self) -> Dict:
        """Build the /tracked_person message."""
        active_tracks = [
            t for t in self.tracks if t.frames_missing == 0
        ]
        
        tracked_persons = []
        for track in active_tracks:
            action = self._determine_action(
                track.uncertainty, track.similarity
            )
            
            biometrics = {}
            if track.age is not None:
                biometrics['age'] = track.age
            if track.gender is not None:
                biometrics['gender'] = track.gender
            
            tracked_persons.append({
                'track_id': track.track_id,
                'name': track.name,
                'bbox': track.bbox,
                'confidence': round(track.similarity, 4),
                'similarity': round(track.similarity, 4),
                'uncertainty': round(track.uncertainty, 4),
                'biometrics': biometrics,
                'attributes': track.attributes,
                'action': action,
                'frames_seen': track.frames_seen
            })
        
        return {
            'timestamp': int(time.time() * 1000),
            'tracked_persons': tracked_persons
        }
    
    def _visualize(self, image: np.ndarray):
        """Draw tracking visualization."""
        annotated = image.copy()
        
        # Header
        active = [t for t in self.tracks if t.frames_missing == 0]
        cv2.putText(
            annotated, f"Tracking: {len(active)} persons", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        
        # Color map for actions
        action_colors = {
            'GREET': (0, 255, 0),
            'ASK_CLARIFICATION': (0, 255, 255),
            'LEARN': (0, 165, 255),
            'OBSERVE': (128, 128, 128)
        }
        
        for track in active:
            action = self._determine_action(
                track.uncertainty, track.similarity
            )
            color = action_colors.get(action, (255, 255, 255))
            
            x1, y1, x2, y2 = track.bbox
            
            # Draw bbox with track ID
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Action banner
            banner_text = f"ACTION: {action}"
            if track.name != 'Unknown':
                banner_text += f" ({track.name})"
            cv2.putText(
                annotated, banner_text, (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Track info
            info = f"ID:{track.track_id} | {track.name} ({track.similarity:.2f})"
            cv2.putText(
                annotated, info, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )
            
            # Uncertainty bar
            bar_w = x2 - x1
            bar_h = 5
            unc_w = int(bar_w * track.uncertainty)
            cv2.rectangle(
                annotated, (x1, y2 + 5), (x1 + bar_w, y2 + 5 + bar_h),
                (100, 100, 100), -1
            )
            cv2.rectangle(
                annotated, (x1, y2 + 5), (x1 + unc_w, y2 + 5 + bar_h),
                (0, 0, 255), -1
            )
        
        cv2.imshow(self.window_name, annotated)
        cv2.waitKey(1)
    
    def register_identity(self, name: str, image: np.ndarray) -> bool:
        """Register a new identity from an image."""
        if self.identity is None:
            self._log('Identity module not available', level='error')
            return False
        
        success = self.identity.register_from_image(name, image)
        if success:
            self._log(f'Registered identity: {name}')
        else:
            self._log(f'Failed to register identity: {name}', level='warn')
        return success
    
    def shutdown(self):
        """Clean shutdown."""
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        if ROS2_AVAILABLE:
            rclpy.shutdown()


def main(args=None):
    """Entry point for the person tracker node."""
    if not ROS2_AVAILABLE:
        print("=" * 60)
        print("WARNING: ROS 2 not detected — running in standalone mode")
        print("Use tools/run_live_system.py for full pipeline testing")
        print("=" * 60)
        return
    
    rclpy.init(args=args)
    node = PersonTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
