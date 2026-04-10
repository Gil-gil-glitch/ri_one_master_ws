#!/usr/bin/env python3
import sys
import os
import cv2
import time
import numpy as np
import threading
from typing import Dict, Optional, List, Tuple

# Add the src directory to the python path so we can import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach src/person_perception
pkg_dir = os.path.join(script_dir, '..')
sys.path.append(pkg_dir)

from person_perception.core.identity import IdentityRecognizer
from person_perception.core.clip_attributes import ClipAttributeDetector

class StandaloneRunner:
    def __init__(self):
        print("Initializing Active Perception System...")
        
        # Initialize Identity Recognizer (InsightFace - GPU)
        self.identity = IdentityRecognizer()
        
        # Initialize CLIP Attribute Detector (GPU)
        try:
            self.clip = ClipAttributeDetector()
            self.use_clip = True
        except Exception as e:
            print(f"Warning: CLIP init failed ({e}). Attributes will be disabled.")
            self.use_clip = False
        
        # Configuration
        self.UNCERTAINTY_THRESHOLD = 0.4
        self.SIMILARITY_THRESHOLD = 0.65
        
        # State
        self.last_attributes: List[str] = []
        self.last_clip_time = 0
        self.clip_interval = 1.0  # Run CLIP every 1 second
        
        print("System Ready!")

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n" + "="*50)
        print("  ACTIVE PERCEPTION RUNNER (HYBRID CLIP)")
        print("  Press 'Q' to quit")
        print("="*50 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()

            # 1. Get Identity & Uncertainty (Every Frame - 30Hz)
            # Returns: (name, similarity, uncertainty, age, gender)
            # Note: Updated IdentityRecognizer.get_identity returns 5 values now
            try:
                result = self.identity.get_identity(frame)
                if len(result) == 5:
                    name, similarity, uncertainty, age, gender = result
                else:
                    # Fallback for old signature
                    name, similarity, uncertainty = result
                    age, gender = None, None
            except Exception as e:
                # Fallback if method signature mismatch
                print(f"Identity error: {e}")
                name, similarity = "Unknown", 0.0
                uncertainty = 1.0

            # 2. Determine Action (State Machine)
            action = "OBSERVE"
            face_detected = name is not None and name != "Unknown"
            status_color = (100, 100, 100) # Gray
            
            # Helper logic for "No Face"
            if uncertainty == 1.0 and similarity == 0.0:
                action = "OBSERVE"
            else:
                if uncertainty > self.UNCERTAINTY_THRESHOLD:
                    action = "ASK_CLARIFICATION"
                    status_color = (0, 165, 255) # Orange
                elif similarity > self.SIMILARITY_THRESHOLD:
                    action = f"GREET ({name})"
                    status_color = (0, 255, 0) # Green
                else:
                    action = "LEARN"
                    status_color = (0, 0, 255) # Red

            # 3. Get Attributes via CLIP (Every 1 Second)
            if self.use_clip and action != "OBSERVE":
                # Only run if we see a person
                if current_time - self.last_clip_time > self.clip_interval:
                    self.last_clip_time = current_time
                    
                    # We need a bounding box for the person.
                    # InsightFace detected the face, but CLIP needs the body.
                    # Heuristic: Extrapolate body from face box or just use full frame center crop?
                    # Better: IdentityRecognizer usually calculates face bbox.
                    # Let's ask IdentityRecognizer for the bbox or use a heuristic.
                    # For this standalone demo, let's use a "Central Crop" heuristic 
                    # assuming user is roughly centered when interacting.
                    h, w = frame.shape[:2]
                    # Heuristic Body Box: Center 50% width, Top 10% to Bottom 90%
                    x1 = int(w * 0.25)
                    x2 = int(w * 0.75)
                    y1 = int(h * 0.1)
                    y2 = int(h * 0.9)
                    person_box = (x1, y1, x2, y2)
                    
                    try:
                        self.last_attributes = self.clip.detect_attributes(frame, person_box)
                    except Exception as e:
                        print(f"CLIP error: {e}")

            # 4. Draw UI
            self._draw_hud(frame, action, name, similarity, uncertainty, status_color)

            cv2.imshow("Active Perception Standalone", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _draw_hud(self, frame, action, name, similarity, uncertainty, color):
        # Top Bar Background
        cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)
        
        # Action (Large Text)
        cv2.putText(frame, f"ACTION: {action}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Stats (Small Text)
        stats = f"Identity: {name} | Sim: {similarity:.2f} | Unc: {uncertainty:.2f}"
        cv2.putText(frame, stats, (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        # Attributes (Small Text)
        attr_str = " | ".join(self.last_attributes) if self.last_attributes else "Scanning..."
        cv2.putText(frame, f"Attributes: {attr_str}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

if __name__ == "__main__":
    runner = StandaloneRunner()
    runner.run()
