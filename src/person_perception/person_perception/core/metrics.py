"""
Perception Metrics Module
=========================
Core metrics for person perception and interaction decision-making.
Implements confidence scoring and action recommendation logic for RoboCup@Home tasks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import entropy as scipy_entropy


class PerceptionMetrics:
    """
    Calculate confidence metrics and action recommendations
    for person perception and interaction decisions.
    """
    
    # Action recommendation thresholds
    HIGH_CONFIDENCE = 0.8   # Above this -> GREET
    LOW_CONFIDENCE = 0.3    # Below this -> LEARN
    BORDERLINE_LOW = 0.4    # Between LOW and BORDERLINE_HIGH -> ASK_CLARIFICATION
    BORDERLINE_HIGH = 0.6   # Between BORDERLINE_LOW and HIGH -> ASK_CLARIFICATION
    
    # Uncertainty thresholds
    LOW_UNCERTAINTY = 0.2
    HIGH_UNCERTAINTY = 0.7
    
    @staticmethod
    def calculate_uncertainty_heuristic(similarity_score: float) -> float:
        """
        Calculate uncertainty using a heuristic formula.
        
        Maps similarity scores to uncertainty:
        - Score of 0.5 (borderline) -> Uncertainty = 1.0 (Maximum)
        - Score of 0.0 or 1.0 (certain) -> Uncertainty = 0.0 (Minimum)
        
        Formula: uncertainty = 1.0 - (2 * abs(score - 0.5))
        
        Args:
            similarity_score: Cosine similarity score (0-1)
            
        Returns:
            Uncertainty score (0-1), higher means more uncertain
        """
        # Clamp to valid range
        score = max(0.0, min(1.0, similarity_score))
        
        # Heuristic formula
        uncertainty = 1.0 - (2.0 * abs(score - 0.5))
        
        return max(0.0, uncertainty)
    
    @staticmethod
    def calculate_uncertainty_entropy(probabilities: Dict[str, float]) -> float:
        """
        Calculate uncertainty using Shannon entropy.
        
        Higher entropy = more uniform distribution = higher uncertainty.
        Normalized to 0-1 range based on number of classes.
        
        Args:
            probabilities: Dict of {identity: probability} or {identity: similarity_score}
            
        Returns:
            Normalized entropy (0-1), higher means more uncertain
        """
        if not probabilities:
            return 1.0  # Maximum uncertainty if no data
        
        # Convert similarities to probability distribution (softmax-like normalization)
        values = np.array(list(probabilities.values()))
        
        # Shift to positive and apply softmax for proper probability distribution
        values = values - values.min() + 1e-6  # Ensure positive
        probs = values / values.sum()
        
        # Calculate entropy
        raw_entropy = scipy_entropy(probs, base=2)
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log2(len(probs)) if len(probs) > 1 else 1.0
        
        if max_entropy == 0:
            return 0.0
        
        return min(1.0, raw_entropy / max_entropy)
    
    @classmethod
    def calculate_uncertainty(
        cls,
        confidence: float,
        all_scores: Optional[Dict[str, float]] = None,
        method: str = 'hybrid'
    ) -> float:
        """
        Calculate uncertainty using specified method.
        
        Args:
            confidence: Best match confidence/similarity (0-1)
            all_scores: All identity similarity scores (for entropy-based)
            method: 'heuristic', 'entropy', or 'hybrid' (average of both)
            
        Returns:
            Uncertainty score (0-1)
        """
        if method == 'heuristic':
            return cls.calculate_uncertainty_heuristic(confidence)
        
        elif method == 'entropy':
            if all_scores is None or len(all_scores) <= 1:
                # Fall back to heuristic if not enough data
                return cls.calculate_uncertainty_heuristic(confidence)
            return cls.calculate_uncertainty_entropy(all_scores)
        
        elif method == 'hybrid':
            heuristic = cls.calculate_uncertainty_heuristic(confidence)
            
            if all_scores is not None and len(all_scores) > 1:
                entropy_based = cls.calculate_uncertainty_entropy(all_scores)
                return (heuristic + entropy_based) / 2.0
            
            return heuristic
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'heuristic', 'entropy', or 'hybrid'.")
    
    @classmethod
    def get_action_recommendation(
        cls,
        confidence: float,
        uncertainty: float,
        is_known: bool
    ) -> str:
        """
        Get action recommendation based on confidence and uncertainty.
        
        Actions:
        - GREET: High confidence known person
        - ASK_CLARIFICATION: Borderline confidence, need more information
        - LEARN: Unknown person or very low confidence
        - OBSERVE: Need more data before making a decision
        
        Args:
            confidence: Recognition confidence (0-1)
            uncertainty: Uncertainty score (0-1)
            is_known: Whether the person matched a known identity
            
        Returns:
            Action recommendation string
        """
        # Unknown person -> LEARN
        if not is_known or confidence < cls.LOW_CONFIDENCE:
            return "LEARN"
        
        # High confidence and low uncertainty -> GREET
        if confidence >= cls.HIGH_CONFIDENCE and uncertainty < cls.LOW_UNCERTAINTY:
            return "GREET"
        
        # High uncertainty regardless of confidence -> need clarification
        if uncertainty >= cls.HIGH_UNCERTAINTY:
            return "ASK_CLARIFICATION"
        
        # Borderline confidence -> ASK_CLARIFICATION
        if cls.BORDERLINE_LOW <= confidence <= cls.BORDERLINE_HIGH:
            return "ASK_CLARIFICATION"
        
        # Moderate confidence with moderate uncertainty
        if confidence >= cls.BORDERLINE_HIGH:
            if uncertainty <= cls.LOW_UNCERTAINTY:
                return "GREET"
            else:
                return "ASK_CLARIFICATION"
        
        # Default: observe more
        return "OBSERVE"
    
    @classmethod
    def compute_perception_metrics(
        cls,
        identity_result: Optional[Dict],
        method: str = 'hybrid'
    ) -> Dict:
        """
        Compute full perception metrics from identity recognition result.
        
        Args:
            identity_result: Result from IdentityRecognizer.process_person()
            method: Uncertainty calculation method
            
        Returns:
            Dictionary with all perception metrics
        """
        if identity_result is None:
            return {
                'is_human': True,  # Assumed from YOLO detection
                'face_detected': False,
                'id_label': 'Unknown',
                'confidence': 0.0,
                'uncertainty_score': 1.0,
                'attributes': [],
                'action_recommendation': 'OBSERVE'
            }
        
        confidence = identity_result['confidence']
        all_scores = identity_result.get('all_scores', {})
        is_known = identity_result['id_label'] != 'Unknown'
        
        # Calculate uncertainty
        uncertainty = cls.calculate_uncertainty(confidence, all_scores, method)
        
        # Get action recommendation
        action = cls.get_action_recommendation(confidence, uncertainty, is_known)
        
        # Add color attribute if available
        attributes = identity_result.get('attributes', []).copy()
        
        return {
            'is_human': True,
            'face_detected': True,
            'id_label': identity_result['id_label'],
            'confidence': round(confidence, 4),
            'uncertainty_score': round(uncertainty, 4),
            'attributes': attributes,
            'action_recommendation': action
        }
    
    @staticmethod
    def analyze_uncertainty_distribution(
        uncertainty_history: List[float],
        window_size: int = 10
    ) -> Dict:
        """
        Analyze uncertainty distribution over time for performance insights.
        
        Args:
            uncertainty_history: List of historical uncertainty values
            window_size: Rolling window size for statistics
            
        Returns:
            Statistics about uncertainty dynamics
        """
        if not uncertainty_history:
            return {
                'mean': 0.0,
                'std': 0.0,
                'trend': 'stable',
                'stability_score': 1.0
            }
        
        arr = np.array(uncertainty_history)
        
        # Calculate statistics
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        
        # Trend analysis using recent window
        if len(arr) >= window_size:
            recent = arr[-window_size:]
            first_half = np.mean(recent[:window_size//2])
            second_half = np.mean(recent[window_size//2:])
            
            diff = second_half - first_half
            if diff > 0.1:
                trend = 'increasing'
            elif diff < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        # Stability score (inverse of variance, normalized)
        stability = 1.0 / (1.0 + std * 5)
        
        return {
            'mean': round(mean, 4),
            'std': round(std, 4),
            'trend': trend,
            'stability_score': round(stability, 4)
        }


# Backward compatibility alias
ResearchMetrics = PerceptionMetrics
