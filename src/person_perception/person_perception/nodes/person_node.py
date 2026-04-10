"""
Person Perception ROS 2 Node with Active Perception State Machine
==================================================================
Main ROS 2 node integrating vision, identity, and research modules.
Publishes person perception information to /perception/person_info.

Implements Active Perception with entropy-based uncertainty calculation
and action state machine (GREET, ASK_CLARIFICATION, LEARN, OBSERVE).

NOTE: If ROS 2 is not installed, use tools/run_live_system.py instead.
"""

import os
import json
from typing import Optional, Dict

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
    Node = object  # Placeholder for class inheritance

from ..core.vision import VisionProcessor

# Conditional import for InsightFace
try:
    from ..core.identity import IdentityRecognizer
    IDENTITY_AVAILABLE = True
except ImportError:
    IDENTITY_AVAILABLE = False


class PersonNode(Node):
    """
    ROS 2 Node for person perception with Active Perception state machine.
    
    Pipeline:
    1. RealSense -> Color + Depth frames at 30Hz
    2. YOLOv8 -> Person detection (filter class 0, select largest bbox)
    3. InsightFace -> Face detection + embedding extraction (GPU accelerated)
    4. Identity comparison -> Cosine similarity with known embeddings
    5. State Machine Logic:
       - IF uncertainty > 0.4 (and person detected) -> action = "ASK_CLARIFICATION"
       - ELIF similarity > 0.65 -> action = "GREET"
       - ELSE -> action = "LEARN"
       - No person -> action = "OBSERVE"
    6. Publish -> JSON message to /perception/person_info
    """
    
    # Active Perception thresholds
    UNCERTAINTY_THRESHOLD = 0.4  # Above this -> ASK_CLARIFICATION
    SIMILARITY_THRESHOLD = 0.65  # Above this -> GREET
    
    def __init__(self):
        super().__init__('person_perception_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('embeddings_path', '')
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        conf_threshold = self.get_parameter('conf_threshold').value
        publish_rate = self.get_parameter('publish_rate').value
        self.show_debug = self.get_parameter('show_debug_window').value
        embeddings_path = self.get_parameter('embeddings_path').value
        
        # Initialize vision processor (YOLO wrapper)
        self.get_logger().info('Initializing VisionProcessor (YOLO)...')
        self.vision = VisionProcessor(
            model_path=model_path,
            conf_threshold=conf_threshold
        )
        
        # Initialize identity recognizer (InsightFace with GPU)
        self.identity: Optional[IdentityRecognizer] = None
        if IDENTITY_AVAILABLE:
            try:
                self.get_logger().info('Initializing IdentityRecognizer (InsightFace GPU)...')
                self.identity = IdentityRecognizer(ctx_id=0)  # GPU context
                
                # Load additional embeddings if path specified
                if embeddings_path and os.path.exists(embeddings_path):
                    self.identity.load_embeddings(embeddings_path)
                
                self.get_logger().info(
                    f'Known identities: {self.identity.get_known_identities()}'
                )
            except Exception as e:
                self.get_logger().warn(f'InsightFace initialization failed: {e}')
                self.identity = None
        else:
            self.get_logger().warn(
                'InsightFace not available. Face recognition disabled.'
            )
        
        # Publisher for person info
        self.publisher = self.create_publisher(
            String,
            '/perception/person_info',
            10
        )
        
        # Debug window setup
        if self.show_debug:
            self.window_name = "Person Perception Debug"
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 960, 720)
            except Exception as e:
                self.get_logger().warn(f'OpenCV window init failed: {e}')
                self.show_debug = False
        
        # Timer for processing loop (30Hz)
        self.timer = self.create_timer(1.0 / publish_rate, self.process_callback)
        
        self.get_logger().info('Person Perception Node started with Active Perception!')
    
    def process_callback(self):
        """
        Main processing callback - runs at 30Hz.
        
        Pipeline:
        1. Grab Frame
        2. Detect Person (YOLO)
        3. Get Identity (InsightFace)
        4. State Machine Logic
        5. Publish JSON
        """
        # Get frames from RealSense
        color_image, depth_image = self.vision.get_frames()
        if color_image is None:
            return
        
        # Detect person using YOLO
        person = self.vision.get_closest_person(color_image, depth_image)
        
        # Get identity and determine action
        perception_msg = self._process_perception(person, color_image)
        
        # Publish JSON message
        msg = String()
        msg.data = json.dumps(perception_msg)
        self.publisher.publish(msg)
        
        # Debug visualization
        if self.show_debug:
            self._visualize_debug(color_image, person, perception_msg)
    
    def _process_perception(
        self,
        person: Optional[Dict],
        color_image: np.ndarray
    ) -> Dict:
        """
        Process perception and apply Active Perception state machine.
        
        NLP Database Schema:
        {
            "is_human": true,
            "id": "Jonathan",           // Primary Key for DB
            "uncertainty": 0.1,         // Research Metric
            "attributes": ["Male", "22", "Red_Shirt", "Backpack"],
            "action": "GREET"           // Decision State
        }
        
        Args:
            person: Person detection dict from VisionProcessor (YOLO)
            color_image: Original color image
            
        Returns:
            Perception message dictionary matching NLP database schema
        """
        # No person detected
        if person is None:
            return {
                'is_human': False,
                'id': None,
                'uncertainty': 1.0,
                'attributes': [],
                'action': 'OBSERVE'
            }
        
        # Default values
        identity_name = 'Unknown'
        similarity_score = 0.0
        uncertainty_score = 1.0
        age = None
        gender = None
        
        # Get identity + biometrics using InsightFace
        if self.identity is not None:
            try:
                # Get identity using the Active Perception method (5-tuple)
                identity_name, similarity_score, uncertainty_score, age, gender = \
                    self.identity.get_identity(color_image)
            except Exception as e:
                self.get_logger().warn(f'Identity processing error: {e}')
        
        # Get clothing/accessories attributes from VisionProcessor
        clothing_attributes = []
        try:
            clothing_attributes = self.vision.detect_attributes(
                color_image, 
                person['bbox']
            )
        except Exception as e:
            self.get_logger().warn(f'Attribute detection error: {e}')
        
        # Build combined attributes list: [gender, age, clothing, accessories]
        attributes = []
        if gender:
            attributes.append(gender)
        if age is not None:
            attributes.append(str(age))
        attributes.extend(clothing_attributes)
        
        # Apply State Machine Logic
        action = self._determine_action(
            person_detected=True,
            uncertainty=uncertainty_score,
            similarity=similarity_score
        )
        
        # Build response matching NLP database schema
        # CRUCIAL: Keep is_human=True even for Unknown (so NLP can ask clarifying questions)
        return {
            'is_human': True,
            'id': identity_name if identity_name != 'Unknown' else 'Unknown',
            'uncertainty': round(float(uncertainty_score), 3),
            'attributes': attributes,
            'action': action
        }
    
    def _determine_action(
        self,
        person_detected: bool,
        uncertainty: float,
        similarity: float
    ) -> str:
        """
        Determine action based on Active Perception state machine.
        
        Decision Logic:
        1. IF uncertainty > 0.4 (and person detected) -> "ASK_CLARIFICATION"
        2. ELIF similarity > 0.65 -> "GREET"
        3. ELSE -> "LEARN"
        
        Args:
            person_detected: Whether a person was detected
            uncertainty: Uncertainty score (0-1)
            similarity: Similarity score (0-1)
            
        Returns:
            Action string: "GREET", "ASK_CLARIFICATION", "LEARN", or "OBSERVE"
        """
        if not person_detected:
            return 'OBSERVE'
        
        # State machine logic as specified
        if uncertainty > self.UNCERTAINTY_THRESHOLD:
            return 'ASK_CLARIFICATION'
        elif similarity > self.SIMILARITY_THRESHOLD:
            return 'GREET'
        else:
            return 'LEARN'
    
    def _visualize_debug(
        self,
        color_image: np.ndarray,
        person: Optional[Dict],
        perception_msg: Dict
    ):
        """Draw debug visualization."""
        annotated = color_image.copy()
        
        # Draw action banner at top
        action = perception_msg['action']
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
        
        # Draw identity info
        identity = perception_msg.get('id', 'Unknown') or 'Unknown'
        uncertainty = perception_msg.get('uncertainty', 1.0)
        cv2.putText(
            annotated, f"ID: {identity} | Uncertainty: {uncertainty:.2f}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        if person:
            # Draw person bbox
            x1, y1, x2, y2 = person['bbox']
            color = action_color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label on bbox
            label = f"{identity} ({1.0 - uncertainty:.2f})"
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        # Show and handle ESC
        cv2.imshow(self.window_name, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            self.get_logger().info('ESC pressed. Shutting down...')
            self.shutdown()
    
    def register_identity(self, name: str, image: np.ndarray) -> bool:
        """
        Register a new identity from an image.
        
        Args:
            name: Person's name
            image: BGR image containing their face
            
        Returns:
            True if successful
        """
        if self.identity is None:
            self.get_logger().error('Identity module not available')
            return False
        
        success = self.identity.register_from_image(name, image)
        if success:
            self.get_logger().info(f'Registered identity: {name}')
        else:
            self.get_logger().warn(f'Failed to register identity: {name}')
        return success
    
    def shutdown(self):
        """Clean shutdown of the node."""
        try:
            self.timer.cancel()
        except Exception:
            pass
        
        try:
            if hasattr(self.vision, 'pipeline'):
                self.vision.pipeline.stop()
        except Exception:
            pass
        
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        if ROS2_AVAILABLE:
            rclpy.shutdown()


def main(args=None):
    """Entry point for the person perception node."""
    if not ROS2_AVAILABLE:
        print("=" * 60)
        print("WARNING: ROS 2 not detected!")
        print("=" * 60)
        print("ROS 2 (rclpy) is not installed in this environment.")
        print("")
        print("Please run the standalone simulator instead:")
        print("  python tools/run_live_system.py")
        print("")
        print("Or install ROS 2 and source your workspace:")
        print("  source /opt/ros/humble/setup.bash")
        print("=" * 60)
        return
    
    rclpy.init(args=args)
    
    node = PersonNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
