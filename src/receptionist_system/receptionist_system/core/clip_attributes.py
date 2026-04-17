import torch
import clip
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

class ClipAttributeDetector:
    """
    Detects attributes (accessories, clothing colors) using OpenAI CLIP.
    
    This is designed to be run at a lower frequency (e.g., 1Hz) than the main
    perception loop due to the computational cost of the Vision Transformer.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize CLIP model.
        
        Args:
            model_name: CLIP model variant (default: ViT-B/32)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[CLIP] Loading model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Pre-defined attribute prompts
        # Format: Category -> (Positive Prompts, Negative Prompts or Alternatives)
        # Note: CLIP works best when comparing probabilities across a set of diverse descriptions.
        self.attribute_prompts = {
            "Eyewear": [
                "a photo of a person wearing glasses",
                "a photo of a person without glasses"
            ],
            "Headwear": [
                "a photo of a person wearing a hat",
                "a photo of a person without a hat"
            ],
            "Scarf": [
                "a photo of a person wearing a scarf",
                "a photo of a person without a scarf"
            ],
            "ShirtColor": [
                "a person wearing a red shirt",
                "a person wearing a blue shirt",
                "a person wearing a green shirt",
                "a person wearing a black shirt",
                "a person wearing a white shirt",
                "a person wearing a grey shirt",
                "a person wearing a yellow shirt",
                "a person wearing a pink shirt",
                "a person wearing a purple shirt"
            ],
            "HairColor": [
                "a person with black hair",
                "a person with brown hair",
                "a person with blonde hair",
                "a person with red hair",
                "a person with grey hair",
                "a person with white hair",
                "a person who is bald"
            ],
            "Outerwear": [
                "a person wearing a t-shirt",
                "a person wearing a sweater",
                "a person wearing a hoodie",
                "a person wearing a jacket",
                "a person wearing a coat",
                "a person wearing a suit"
            ],
            "Jewelry": [
                "a close-up photo of a person wearing earrings",
                "a close-up photo of a person wearing a necklace",
                "a photo of a person without jewelry"
            ]
        }
        
        # Cache for encoded text features
        self.text_features = {}
        self._precompute_text_features()
        print("[CLIP] Initialization complete.")

    def _precompute_text_features(self):
        """Pre-compute text embeddings for all prompts to save runtime."""
        print("[CLIP] Encoding text prompts...")
        for category, prompts in self.attribute_prompts.items():
            text_tokens = clip.tokenize(prompts).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(text_tokens)
                features /= features.norm(dim=-1, keepdim=True)
                self.text_features[category] = (prompts, features)

    def detect_attributes(self, image: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> List[str]:
        """
        Detect attributes for a specific person in the image.
        
        Args:
            image: Full BGR image
            person_bbox: (x1, y1, x2, y2)
            
        Returns:
            List of detected attribute strings (e.g., ["Glasses", "Red Shirt"])
        """
        # Crop person
        x1, y1, x2, y2 = person_bbox
        h, w = image.shape[:2]
        
        # Add a small margin to context, but clamp to image
        margin = int(min(x2-x1, y2-y1) * 0.1)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        if x2 <= x1 or y2 <= y1:
            return []
            
        person_crop = image[y1:y2, x1:x2]
        
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Preprocess and run CLIP
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        detected_attributes = []
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Check Eyewear
            probs = self._get_probs(image_features, "Eyewear")
            if probs[0] > 0.8: # "wearing glasses" vs "without"
                detected_attributes.append("Glasses")
                
            # Check Headwear
            probs = self._get_probs(image_features, "Headwear")
            if probs[0] > 0.75:
                detected_attributes.append("Hat")

            # Check Scarf
            probs = self._get_probs(image_features, "Scarf")
            if probs[0] > 0.75:
                detected_attributes.append("Scarf")
                
            # Check Outerwear/Clothing Type
            probs_outer = self._get_probs(image_features, "Outerwear")
            outer_idx = np.argmax(probs_outer)
            outer_type = self.attribute_prompts["Outerwear"][outer_idx].replace("a person wearing a ", "").capitalize()
            
            # Check Clothing Color
            probs_color = self._get_probs(image_features, "ShirtColor")
            color_idx = np.argmax(probs_color)
            color_prompt = self.attribute_prompts["ShirtColor"][color_idx]
            color_name = color_prompt.split("wearing a ")[1].split(" ")[0].capitalize()
            
            # Combine Color + Type
            # If color confidence is decent (>0.3), prepend it.
            # otherwise just return the type.
            if probs_color[color_idx] > 0.35:
                detected_attributes.append(f"{color_name} {outer_type}")
            else:
                detected_attributes.append(outer_type)
            
            # Check Hair Color
            probs = self._get_probs(image_features, "HairColor")
            best_idx = np.argmax(probs)
            hair_prompt = self.attribute_prompts["HairColor"][best_idx]
            if "bald" in hair_prompt:
                 detected_attributes.append("Bald")
            else:
                hair_color = hair_prompt.split("with ")[1].split(" ")[0].capitalize()
                detected_attributes.append(f"{hair_color} Hair")

            # Check Jewelry (necklace/earrings)
            probs = self._get_probs(image_features, "Jewelry")
            # 0: Earrings, 1: Necklace, 2: None
            if probs[0] > 0.3: # Earrings
                 detected_attributes.append("Earrings")
            if probs[1] > 0.4: # Necklace (easier to see than earrings)
                 detected_attributes.append("Necklace")

        return detected_attributes

    def _get_probs(self, image_features, category: str) -> np.ndarray:
        """Calculate softmax probabilities for a category."""
        prompts, text_embeds = self.text_features[category]
        
        # similarity = (100.0 * image @ text.T).softmax(dim=-1)
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_embeds.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        return probs
