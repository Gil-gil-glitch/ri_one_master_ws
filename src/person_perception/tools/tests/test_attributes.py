import cv2
import sys
import os
import numpy as np
import time

# Add parent path for imports
# Add all src packages to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
for pkg in os.listdir(src_dir):
    pkg_path = os.path.join(src_dir, pkg)
    if os.path.isdir(pkg_path):
        sys.path.append(pkg_path)

from ultralytics import YOLO
from receptionist_system.core.clip_attributes import ClipAttributeDetector

def main():
    print("Initializing Vision High-Detail Test (YOLO + YOLO-Pose + CLIP)...")
    try:
        clip = ClipAttributeDetector()
        yolo_det = YOLO('yolov8s-worldv2.pt')
        yolo_det.set_classes(["person"])
        
        # YOLO-Pose for precision landmarks
        yolo_pose = YOLO('yolov8n-pose.pt')
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    print("Press 'q' to quit. Test: Wear a hat/glasses or change shirts.")

    last_attrs = {}
    last_run = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detection Pass
        results_det = yolo_det(frame, conf=0.4, verbose=False)
        person_box = None
        
        # Find largest person
        max_area = 0
        for box in results_det[0].boxes:
            b = box.xyxy[0].cpu().numpy()
            area = (b[2]-b[0]) * (b[3]-b[1])
            if area > max_area:
                max_area = area
                person_box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

        # Fallback to center if no one found (for initialization)
        if person_box is None:
            h, w = frame.shape[:2]
            person_box = (int(w*0.2), int(h*0.1), int(w*0.8), int(h*0.9))

        # 2. Keypoint Pass (YOLO Pose)
        landmarks = None
        results_pose = yolo_pose(frame, verbose=False)
        if len(results_pose) > 0 and results_pose[0].keypoints is not None:
             kp = results_pose[0].keypoints.xyn[0].cpu().numpy() # Normalized
             if len(kp) > 6:
                landmarks = {
                    'left_eye': kp[1].tolist(),
                    'right_eye': kp[2].tolist(),
                    'left_ear': kp[3].tolist(),
                    'right_ear': kp[4].tolist(),
                    'left_shoulder': kp[5].tolist(),
                    'right_shoulder': kp[6].tolist()
                }

        # 3. Attribute Pass
        now = time.time()
        if now - last_run > 0.8: # Slightly faster 1.2Hz
            last_run = now
            last_attrs = clip.detect_attributes(frame, person_box, landmarks=landmarks, include_debug=True)
            # Print for debugging
            print("-" * 30)
            for attr, score in last_attrs.items():
                if attr == "_debug": continue
                if isinstance(score, float):
                    print(f"  {attr:<15} | Score: {score:.4f}")
                else:
                    print(f"  {attr:<15} | Value: {score}")


        # Draw ROIs
        if "_debug" in last_attrs:
            db = last_attrs["_debug"]
            for name, box in db.get("active_regions", {}).items():
                # Specialized Colors for Pinpoint regions
                if name == "Glasses": color = (0, 255, 0) # Green for eyes
                elif name == "Earrings": color = (255, 0, 255) # Magenta for ears
                elif name == "Clothing": color = (255, 255, 0) # Cyan for body
                else: color = (0, 255, 255) # Yellow for head
                
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, name, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Display attributes
        y0, dy = 100, 30
        for i, (attr, score) in enumerate(last_attrs.items()):
            if attr == "_debug": continue
            if ">>" in attr:
                label = f"{attr} {score}"
            elif isinstance(score, float):
                label = f"{attr} ({score:.2f})"
            else:
                label = f"{attr}: {score}"
            cv2.putText(frame, label, (10, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        cv2.imshow("Test: CLIP Attribute Extraction", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
