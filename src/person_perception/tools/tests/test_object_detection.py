import cv2
import sys
import os
import numpy as np

# Add parent path for imports
# Add all src packages to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
for pkg in os.listdir(src_dir):
    pkg_path = os.path.join(src_dir, pkg)
    if os.path.isdir(pkg_path):
        sys.path.insert(0, pkg_path)

from ultralytics import YOLO

def main():
    print("Initializing YOLO-World v2 Object Detection Test...")
    model = YOLO('yolov8s-worldv2.pt')
    
    # Competition objects
    prompts = ["paper bag", "chair", "person"]
    model.set_classes(prompts)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit. Test: Hold up a paper bag or sit on a chair.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()
        
        cv2.imshow("Test: Zero-Shot Object Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
