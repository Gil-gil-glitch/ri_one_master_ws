"""
Identity Module: InsightFace Wrapper for Face Recognition with Active Perception
=================================================================================
Provides face detection, embedding extraction, and identity comparison
using InsightFace library with 512-dimensional face embeddings.

Implements "Active Perception" with entropy-based uncertainty calculation.
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from scipy.spatial.distance import cosine

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class IdentityRecognizer:
    """
    Face recognition using InsightFace for embedding extraction
    and cosine similarity for identity matching.
    
    Implements Active Perception with uncertainty-based decision making.
    """
    
    # Recognition thresholds
    RECOGNITION_THRESHOLD = 0.5  # Minimum similarity to consider a match
    HIGH_CONFIDENCE_THRESHOLD = 0.65  # High confidence match for GREET action
    UNCERTAINTY_THRESHOLD = 0.4  # Above this, ASK_CLARIFICATION is triggered
    
    def __init__(
        self,
        model_name: str = 'buffalo_l',
        ctx_id: int = 0,
        det_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize InsightFace FaceAnalysis with CUDA GPU acceleration.
        
        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_s, etc.)
            ctx_id: GPU context ID (0 for first GPU, -1 for CPU)
            det_size: Detection input size
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "InsightFace is not installed. "
                "Please install with: pip install insightface onnxruntime-gpu"
            )
        
        # Initialize with CUDA provider for GPU acceleration
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        
        # Known embeddings database: {name: embedding}
        self._known_embeddings: Dict[str, np.ndarray] = {}
        
        # Initialize with hardcoded known faces (mock if file missing)
        self.known_faces: Dict[str, np.ndarray] = {}
        self._load_hardcoded_embeddings()
    
    def _load_hardcoded_embeddings(self) -> None:
        """
        Load known face embeddings from the data/faces directory.
        Auto-discovers all *_embed.npy files, deriving names from filenames.
        E.g., 'jonathan_embed.npy' -> 'Jonathan'
        """
        # Get the path to the data/faces directory relative to this module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(module_dir, '..', '..', 'data', 'faces')
        
        # Override to use the actual source directory so newly registered faces are found immediately
        src_data_dir = os.path.expanduser('~/ros2_ws/src/person_perception/data/faces')
        if os.path.isdir(src_data_dir):
            data_dir = src_data_dir
        
        if not os.path.isdir(data_dir):
            return
        
        # Auto-discover all *_embed.npy files
        import glob
        embed_files = glob.glob(os.path.join(data_dir, "*_embed.npy"))
        
        for embed_path in embed_files:
            filename = os.path.basename(embed_path)
            # Derive name: "jonathan_embed.npy" -> "Jonathan"
            name = filename.replace("_embed.npy", "").replace("_", " ").title()
            try:
                embedding = np.load(embed_path)
                embedding = embedding / np.linalg.norm(embedding)
                self.known_faces[name] = embedding
                self._known_embeddings[name] = embedding
            except Exception as e:
                pass
    
    def register_identity(self, name: str, embedding: np.ndarray) -> None:
        """
        Register a known identity with their face embedding.
        
        Args:
            name: Person's name/identifier
            embedding: 512-dimensional face embedding
        """
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        self._known_embeddings[name] = embedding
        self.known_faces[name] = embedding
    
    def register_from_image(self, name: str, image: np.ndarray) -> bool:
        """
        Register a known identity from an image.
        
        Args:
            name: Person's name/identifier
            image: BGR image containing the face
            
        Returns:
            True if registration successful, False otherwise
        """
        faces = self.detect_faces(image)
        if not faces:
            return False
        
        # Use the largest face
        largest_face = max(faces, key=lambda f: f['area'])
        self.register_identity(name, largest_face['embedding'])
        return True
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces and extract embeddings from image.
        
        Args:
            image: BGR image to process
            
        Returns:
            List of face dictionaries with:
            - bbox: (x1, y1, x2, y2)
            - embedding: 512-dimensional face embedding
            - age: estimated age
            - gender: 'Male' or 'Female'
            - area: bounding box area
            - det_score: detection confidence
        """
        faces = self.app.get(image)
        
        result = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            area = (x2 - x1) * (y2 - y1)
            
            # Get attributes
            age = int(face.age) if hasattr(face, 'age') else None
            gender = 'Male' if face.gender == 1 else 'Female' if hasattr(face, 'gender') else None
            
            result.append({
                'bbox': (x1, y1, x2, y2),
                'embedding': face.embedding,
                'age': age,
                'gender': gender,
                'area': area,
                'det_score': float(face.det_score) if hasattr(face, 'det_score') else None
            })
        
        return result
    
    def _compute_uncertainty(self, similarity_score: float) -> float:
        """
        Calculate uncertainty using the Active Perception entropy-based formula.
        
        Formula: uncertainty = 1.0 - (2 * abs(similarity_score - 0.5))
        
        Interpretation:
        - If score is 0.5, we are 100% uncertain (uncertainty = 1.0)
        - If score is 1.0 or 0.0, we are 0% uncertain (uncertainty = 0.0)
        - Scores near 0.5 represent high ambiguity
        
        Args:
            similarity_score: Cosine similarity score (0-1)
            
        Returns:
            Uncertainty score (0-1), where 1.0 = maximum uncertainty
        """
        return 1.0 - (2.0 * abs(similarity_score - 0.5))
    
    def get_identity(self, image: np.ndarray) -> Tuple[str, float, float, Optional[int], Optional[str]]:
        """
        Main Active Perception method to identify a person in an image.
        
        Uses InsightFace to detect faces, picks the largest face,
        compares against known embeddings, and computes uncertainty.
        
        Args:
            image: BGR image to process
            
        Returns:
            Tuple of (name, similarity_score, uncertainty_score, age, gender)
            - name: Identified name or 'Unknown'
            - similarity_score: Best cosine similarity (0-1)
            - uncertainty_score: Entropy-based uncertainty (0-1)
            - age: Estimated age (int) or None
            - gender: 'Male' or 'Female' or None
        """
        # Detect faces using InsightFace
        faces = self.app.get(image)
        
        if not faces:
            # No face detected - maximum uncertainty
            return ('Unknown', 0.0, 1.0, None, None)
        
        # Pick the largest face (closest/most prominent)
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Extract biometrics
        age = int(largest_face.age) if hasattr(largest_face, 'age') else None
        gender = 'Male' if largest_face.gender == 1 else 'Female' if hasattr(largest_face, 'gender') else None
        
        # Get embedding
        query_embedding = largest_face.embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if not self._known_embeddings:
            # No known embeddings - return unknown with max uncertainty
            return ('Unknown', 0.0, 1.0, age, gender)
        
        # Find best match
        best_name = 'Unknown'
        best_similarity = 0.0
        
        for name, known_emb in self._known_embeddings.items():
            # Cosine similarity = 1 - cosine distance
            similarity = 1.0 - cosine(query_embedding, known_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name
        
        # Apply recognition threshold
        if best_similarity < self.RECOGNITION_THRESHOLD:
            best_name = 'Unknown'
        
        # Calculate uncertainty using entropy-based formula
        uncertainty_score = self._compute_uncertainty(best_similarity)
        
        return (best_name, best_similarity, uncertainty_score, age, gender)
    
    def identify(self, embedding: np.ndarray) -> Dict:
        """
        Identify a face by comparing its embedding to known embeddings.
        
        Args:
            embedding: 512-dimensional face embedding
            
        Returns:
            Dictionary with:
            - id_label: Identified name or 'Unknown'
            - confidence: Cosine similarity score (0-1)
            - uncertainty_score: Entropy-based uncertainty (0-1)
            - all_scores: Dict of scores for all known identities
        """
        if not self._known_embeddings:
            return {
                'id_label': 'Unknown',
                'confidence': 0.0,
                'uncertainty_score': 1.0,
                'all_scores': {}
            }
        
        # Normalize query embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Calculate cosine similarity with all known embeddings
        scores = {}
        for name, known_emb in self._known_embeddings.items():
            # Cosine similarity = 1 - cosine distance
            similarity = 1 - cosine(embedding, known_emb)
            scores[name] = float(similarity)
        
        # Find best match
        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]
        
        # Calculate uncertainty
        uncertainty_score = self._compute_uncertainty(best_score)
        
        # Apply threshold
        if best_score < self.RECOGNITION_THRESHOLD:
            return {
                'id_label': 'Unknown',
                'confidence': best_score,
                'uncertainty_score': uncertainty_score,
                'all_scores': scores
            }
        
        return {
            'id_label': best_name,
            'confidence': best_score,
            'uncertainty_score': uncertainty_score,
            'all_scores': scores
        }
    
    def process_person(
        self,
        image: np.ndarray,
        person_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Dict]:
        """
        Process a person region and return identity information.
        
        Args:
            image: Full BGR image
            person_bbox: Optional (x1, y1, x2, y2) to crop before processing
            
        Returns:
            Dictionary with face info and identity, or None if no face found
        """
        # Crop to person region if specified
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            image = image[y1:y2, x1:x2]
        
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Use the largest face (closest/most prominent)
        face = max(faces, key=lambda f: f['area'])
        
        # Identify
        identity = self.identify(face['embedding'])
        
        # Build attributes list
        attributes = []
        if face['gender']:
            attributes.append(face['gender'])
        if face['age']:
            attributes.append(f"Age_{face['age']}")
        
        return {
            'face_bbox': face['bbox'],
            'id_label': identity['id_label'],
            'confidence': identity['confidence'],
            'uncertainty_score': identity['uncertainty_score'],
            'all_scores': identity['all_scores'],
            'age': face['age'],
            'gender': face['gender'],
            'attributes': attributes,
            'det_score': face['det_score'],
            'embedding': face['embedding']
        }
    
    def get_known_identities(self) -> List[str]:
        """Get list of registered identity names."""
        return list(self._known_embeddings.keys())
    
    def save_embeddings(self, path: str) -> None:
        """Save known embeddings to file."""
        np.savez(path, **self._known_embeddings)
    
    def load_embeddings(self, path: str) -> None:
        """Load known embeddings from file."""
        if os.path.exists(path):
            data = np.load(path)
            self._known_embeddings = {key: data[key] for key in data.files}
            self.known_faces = self._known_embeddings.copy()
