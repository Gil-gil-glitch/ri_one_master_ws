import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO

# Add all src packages to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
for pkg in os.listdir(src_dir):
    pkg_path = os.path.join(src_dir, pkg)
    if os.path.isdir(pkg_path):
        sys.path.insert(0, pkg_path)

def main():
    print("Initializing 3D Pointing Interaction Test (YOLO-Pose)...")

    # Use YOLO-Pose instead of Mediapipe (consistent with our unified vision stack)
    try:
        yolo_pose = YOLO('yolov8n-pose.pt')
    except Exception as e:
        print(f"Failed to load YOLO-Pose model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    print("Press 'q' to quit. Test: Point at something in the room.")

    # COCO keypoint indices:
    # 5=left_shoulder, 6=right_shoulder, 9=left_wrist, 10=right_wrist
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        results = yolo_pose(frame, verbose=False)
        
        if len(results) > 0 and results[0].keypoints is not None:
            kps = results[0].keypoints
            
            # Process each detected person
            for person_idx in range(kps.xy.shape[0]):
                xy = kps.xy[person_idx].cpu().numpy()  # (17, 2) pixel coords
                conf = kps.conf[person_idx].cpu().numpy()  # (17,) confidences
                
                # Right arm: shoulder(6) -> wrist(10)
                r_shoulder = xy[6]
                r_wrist = xy[10]
                r_conf = min(conf[6], conf[10])
                
                # Left arm: shoulder(5) -> wrist(9)
                l_shoulder = xy[5]
                l_wrist = xy[9]
                l_conf = min(conf[5], conf[9])
                
                # Pick the arm with the higher combined confidence
                if r_conf > l_conf and r_conf > 0.5:
                    shoulder, wrist, arm_label = r_shoulder, r_wrist, "R"
                elif l_conf > 0.5:
                    shoulder, wrist, arm_label = l_shoulder, l_wrist, "L"
                else:
                    continue  # No confident arm detected
                
                p1 = (int(shoulder[0]), int(shoulder[1]))
                p2 = (int(wrist[0]), int(wrist[1]))
                
                # Draw pointing vector (shoulder -> wrist)
                cv2.line(frame, p1, p2, (0, 255, 255), 3)
                cv2.circle(frame, p2, 8, (0, 0, 255), -1)  # Wrist endpoint
                cv2.circle(frame, p1, 6, (255, 0, 0), -1)  # Shoulder origin
                
                # Extend the ray past the wrist to show pointing direction
                dx = wrist[0] - shoulder[0]
                dy = wrist[1] - shoulder[1]
                mag = np.sqrt(dx**2 + dy**2) + 1e-6
                
                # Extend ray 2x past the wrist
                ext_x = int(wrist[0] + dx * 2)
                ext_y = int(wrist[1] + dy * 2)
                cv2.line(frame, p2, (ext_x, ext_y), (0, 200, 200), 2, cv2.LINE_AA)
                
                # Show direction info
                cv2.putText(frame, f"[{arm_label}] Dir: [{dx/mag:.2f}, {dy/mag:.2f}]", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Test: 3D Pointing (YOLO-Pose)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
