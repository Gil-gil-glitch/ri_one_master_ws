"""
Face Registration Tool
======================
Standalone Python script to enroll new users into the Identity Recognition system.
Opens webcam, detects faces, and saves 512-dimensional face embeddings as .npy files.

Usage:
    python register_face.py
    
    - Press 'S' to save the current face
    - Press 'Q' to quit
"""

import os
import sys
import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("ERROR: InsightFace not installed.")
    print("Install with: pip install insightface onnxruntime-gpu")
    sys.exit(1)


def main():
    """Main registration loop."""
    print("=" * 60)
    print("  FACE REGISTRATION TOOL")
    print("  Press 'S' to Save Face | Press 'Q' to Quit")
    print("=" * 60)
    
    # Initialize InsightFace with CUDA GPU acceleration
    print("\n[INFO] Initializing InsightFace (GPU)...")
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[INFO] InsightFace initialized successfully!")
    except Exception as e:
        print(f"[WARN] GPU init failed, falling back to CPU: {e}")
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'faces')
    os.makedirs(data_dir, exist_ok=True)
    print(f"[INFO] Embeddings will be saved to: {os.path.abspath(data_dir)}")
    
    # Open webcam
    print("[INFO] Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        sys.exit(1)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[INFO] Webcam opened successfully!")
    print("\n>>> Look at the camera. Press 'S' when ready to save your face.\n")
    
    current_embedding = None
    current_bbox = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            continue
        
        # Detect faces
        faces = app.get(frame)
        
        # Find largest face
        current_embedding = None
        current_bbox = None
        
        if faces:
            # Pick the largest face (by area)
            largest_face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            
            x1, y1, x2, y2 = map(int, largest_face.bbox)
            current_bbox = (x1, y1, x2, y2)
            current_embedding = largest_face.embedding
            
            # Draw GREEN bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw label
            cv2.putText(
                frame, "Face Detected", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Draw instructions
        cv2.putText(
            frame, "Press 'S' to Save Face, 'Q' to Quit", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Show status
        status = "FACE DETECTED" if current_embedding is not None else "NO FACE"
        color = (0, 255, 0) if current_embedding is not None else (0, 0, 255)
        cv2.putText(
            frame, f"Status: {status}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        
        # Display frame
        cv2.imshow("Face Registration Tool", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n[INFO] Quitting...")
            break
        
        elif key == ord('s') or key == ord('S'):
            if current_embedding is None:
                print("\n[WARN] No face detected! Please position your face in the camera.")
                continue
            
            # Minimize OpenCV window to show console
            cv2.destroyAllWindows()
            
            # Ask for name
            print("\n" + "=" * 40)
            name = input("Enter your name (e.g., Bob): ").strip()
            
            if not name:
                print("[WARN] Empty name. Cancelled.")
                # Reopen window
                continue
            
            # Normalize and save embedding
            embedding = current_embedding / np.linalg.norm(current_embedding)
            
            # Create filename (lowercase with underscores)
            safe_name = name.lower().replace(" ", "_")
            filename = f"{safe_name}_embed.npy"
            filepath = os.path.join(data_dir, filename)
            
            np.save(filepath, embedding)
            
            print(f"\nSuccess! Saved '{name}' to file:")
            print(f"   {os.path.abspath(filepath)}")
            print(f"   Embedding shape: {embedding.shape}")
            print("=" * 40)
            print("\nYou can register another face or press 'Q' to quit.\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n[INFO] Registration tool closed.")
    print("[INFO] Don't forget to update identity.py to load your new embedding!")


if __name__ == '__main__':
    main()
