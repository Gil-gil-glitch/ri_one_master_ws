"""
Multi-Modal Person Registration Tool
=====================================
Enrolls new users with a 3-layer identity profile:
  1. Face Embedding (Multi-Pose Mean) — InsightFace
  2. Structural Build (Body Proportions) — Mediapipe Pose
  3. Appearance Snapshot (Clothing)      — saved as crop for CLIP

Guided 5-pose flow:  FRONT -> LEFT -> RIGHT -> UP -> DOWN
Saves:
  - {name}_embed.npy        — Mean face embedding (512-d)
  - {name}_struct.npy       — Structural proportions vector
  - {name}_appearance.jpg   — Full-body crop for CLIP (session)

Usage:
    python register_face.py
    - Follow the on-screen pose prompts
    - Press 'S' to capture each pose
    - Press 'Q' to quit at any time
"""

import os
import sys
import json
import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("ERROR: InsightFace not installed.")
    print("Install with: pip install insightface onnxruntime-gpu")
    sys.exit(1)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[WARN] Mediapipe not installed. Structural build capture disabled.")
    MEDIAPIPE_AVAILABLE = False


# ── Pose definitions ──────────────────────────────────────────────
POSE_SEQUENCE = [
    {"name": "FRONT",  "instruction": "Look straight at the camera"},
    {"name": "LEFT",   "instruction": "Turn your head LEFT (show right ear)"},
    {"name": "RIGHT",  "instruction": "Turn your head RIGHT (show left ear)"},
    {"name": "UP",     "instruction": "Tilt your head UP slightly"},
    {"name": "DOWN",   "instruction": "Tilt your head DOWN slightly"},
]


def extract_structural_features(pose_landmarks, image_height: int) -> dict | None:
    """
    Extract stable body proportions from Mediapipe Pose landmarks.
    These ratios are invariant to distance from the camera.

    Returns a dict with named proportions and a flat numpy vector.
    """
    lm = pose_landmarks.landmark

    # Key landmark indices (Mediapipe Pose)
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    NOSE = 0

    def dist(a, b):
        return np.sqrt(
            (lm[a].x - lm[b].x) ** 2
            + (lm[a].y - lm[b].y) ** 2
        )

    shoulder_width = dist(L_SHOULDER, R_SHOULDER)
    torso_length = (dist(L_SHOULDER, L_HIP) + dist(R_SHOULDER, R_HIP)) / 2
    left_arm = dist(L_SHOULDER, L_ELBOW) + dist(L_ELBOW, L_WRIST)
    right_arm = dist(R_SHOULDER, R_ELBOW) + dist(R_ELBOW, R_WRIST)
    hip_width = dist(L_HIP, R_HIP)
    head_to_shoulder = dist(NOSE, L_SHOULDER)

    # Normalise everything relative to torso_length (scale-invariant)
    if torso_length < 1e-6:
        return None

    proportions = {
        "shoulder_to_torso": shoulder_width / torso_length,
        "hip_to_torso": hip_width / torso_length,
        "left_arm_to_torso": left_arm / torso_length,
        "right_arm_to_torso": right_arm / torso_length,
        "head_to_shoulder_ratio": head_to_shoulder / torso_length,
    }

    vec = np.array(list(proportions.values()), dtype=np.float32)
    return {"proportions": proportions, "vector": vec}


def main():
    """Main multi-modal registration loop."""
    print("=" * 60)
    print("  MULTI-MODAL PERSON REGISTRATION TOOL")
    print("  5-Pose Face + Structural Build + Appearance")
    print("=" * 60)

    # ── Ask for name up-front ─────────────────────────────────────
    name = input("\nEnter name to register (e.g., Bob): ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        sys.exit(1)

    safe_name = name.lower().replace(" ", "_")

    # ── Init InsightFace ──────────────────────────────────────────
    print("\n[INFO] Initializing InsightFace (GPU)...")
    try:
        face_app = FaceAnalysis(providers=['CUDAExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[INFO] InsightFace ready (CUDA).")
    except Exception as e:
        print(f"[WARN] GPU failed, falling back to CPU: {e}")
        face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))

    # ── Init Mediapipe Pose ───────────────────────────────────────
    mp_pose = None
    pose_detector = None
    if MEDIAPIPE_AVAILABLE:
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        print("[INFO] Mediapipe Pose ready.")

    # ── Output directory ──────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'faces')
    os.makedirs(data_dir, exist_ok=True)
    print(f"[INFO] Data dir: {os.path.abspath(data_dir)}")

    # ── Open webcam ───────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ─────────────────────────────────────────────────────
    pose_embeddings = []           # collected face embeddings per pose
    structural_vectors = []        # collected body proportion vectors
    appearance_frame = None        # full-body crop for CLIP
    pose_idx = 0                   # current pose in POSE_SEQUENCE
    total_poses = len(POSE_SEQUENCE)

    print(f"\n>>> Registering '{name}' — {total_poses} poses required.\n")

    while pose_idx < total_poses:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        current_pose = POSE_SEQUENCE[pose_idx]

        # ── Face detection ────────────────────────────────────────
        faces = face_app.get(frame)
        face_detected = False
        current_embedding = None

        if faces:
            largest = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            x1, y1, x2, y2 = map(int, largest.bbox)
            current_embedding = largest.embedding
            face_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── Pose detection (structural) ───────────────────────────
        pose_results = None
        if pose_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose_detector.process(rgb)

            # Draw skeleton overlay
            if pose_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(255, 200, 0), thickness=2, circle_radius=2
                    ),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 200, 255), thickness=2
                    ),
                )

        # ── HUD ───────────────────────────────────────────────────
        progress = f"Pose {pose_idx + 1}/{total_poses}: {current_pose['name']}"
        cv2.putText(
            frame, progress, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        cv2.putText(
            frame, current_pose["instruction"], (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        status_text = "FACE OK — Press 'S' to capture" if face_detected else "NO FACE"
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(
            frame, status_text, (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
        )

        cv2.imshow("Multi-Modal Registration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("\n[INFO] Registration cancelled by user.")
            break

        elif key == ord('s') or key == ord('S'):
            if current_embedding is None:
                print(f"[WARN] No face for {current_pose['name']}. Try again.")
                continue

            # Save face embedding for this pose
            emb_norm = current_embedding / np.linalg.norm(current_embedding)
            pose_embeddings.append(emb_norm)
            print(f"  [OK] Captured {current_pose['name']} face embedding.")

            # Save structural vector (only need one good sample)
            if (
                pose_detector is not None
                and pose_results is not None
                and pose_results.pose_landmarks
                and len(structural_vectors) == 0
            ):
                struct = extract_structural_features(
                    pose_results.pose_landmarks, h
                )
                if struct is not None:
                    structural_vectors.append(struct["vector"])
                    print(f"  [OK] Captured structural build.")

            # Save appearance frame on FRONT pose
            if pose_idx == 0:
                appearance_frame = frame.copy()

            pose_idx += 1

    # ── Finalise ──────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    if len(pose_embeddings) == 0:
        print("\n[ERROR] No poses captured. Registration aborted.")
        sys.exit(1)

    # 1. Mean Face Embedding
    mean_embedding = np.mean(pose_embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
    embed_path = os.path.join(data_dir, f"{safe_name}_embed.npy")
    np.save(embed_path, mean_embedding)

    # 2. Structural Build
    struct_path = None
    if structural_vectors:
        struct_vec = np.mean(structural_vectors, axis=0)
        struct_path = os.path.join(data_dir, f"{safe_name}_struct.npy")
        np.save(struct_path, struct_vec)

    # 3. Appearance Snapshot
    appear_path = None
    if appearance_frame is not None:
        appear_path = os.path.join(data_dir, f"{safe_name}_appearance.jpg")
        cv2.imwrite(appear_path, appearance_frame)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  REGISTRATION COMPLETE: {name}")
    print(f"  Poses captured:     {len(pose_embeddings)}/{total_poses}")
    print(f"  Mean embedding:     {os.path.abspath(embed_path)}")
    if struct_path:
        print(f"  Structural build:   {os.path.abspath(struct_path)}")
    if appear_path:
        print(f"  Appearance snap:    {os.path.abspath(appear_path)}")
    print("=" * 60)

    # Save metadata
    meta = {
        "name": name,
        "poses_captured": len(pose_embeddings),
        "has_structural": struct_path is not None,
        "has_appearance": appear_path is not None,
    }
    meta_path = os.path.join(data_dir, f"{safe_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata:           {os.path.abspath(meta_path)}")


if __name__ == '__main__':
    main()
