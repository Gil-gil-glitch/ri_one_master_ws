import cv2
import sys
import os
import numpy as np
import time
import importlib.util

# Add all src packages to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
for pkg in os.listdir(src_dir):
    pkg_path = os.path.join(src_dir, pkg)
    if os.path.isdir(pkg_path):
        sys.path.insert(0, pkg_path)

# Direct file import to bypass core/__init__.py which pulls in pyrealsense2
_identity_path = os.path.join(src_dir, 'person_perception', 'person_perception', 'core', 'identity.py')
_spec = importlib.util.spec_from_file_location("identity", _identity_path)
_identity_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_identity_mod)
IdentityRecognizer = _identity_mod.IdentityRecognizer

def main():
    print("Initializing Identity Lock Test (Phase 2)...")
    try:
        identity = IdentityRecognizer(ctx_id=0)
    except Exception as e:
        print(f"Failed to load InsightFace: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    print("Press 'q' to quit. Test: Walk around or turn your head to test ID persistence.")
    print(f"Known identities: {identity.get_known_identities()}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Get identity (Multi-modal)
        result = identity.get_identity(frame)
        name, sim, unc, age, gender = result
        
        # Visualization
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.putText(frame, f"ID: {name} | Sim: {sim:.2f}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Age: {age} | Gender: {gender}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Test: Identity Lock (InsightFace)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
