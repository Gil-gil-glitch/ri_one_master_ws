import torch
import clip
from PIL import Image
import numpy as np
import cv2
import time
from typing import List, Dict, Optional, Tuple, Any, Union

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
            
        # Temporal Smoothing State
        self.history = {} # track_id -> { category -> [(timestamp, value), ...] }
        self.locked_attributes = {} # track_id -> { category -> locked_value }
            
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
            "Mask": [
                "a close-up photo of a person wearing a face mask over their mouth",
                "a close-up photo of a person with no face mask"
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
            ],
            "Gender": [
                "a photo of a man",
                "a photo of a woman"
            ],
            "Age": [
                "a photo of a child",
                "a photo of a teenager",
                "a photo of an adult",
                "a photo of a senior"
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

    def detect_attributes(self, image: np.ndarray, person_bbox: Tuple[int, int, int, int], 
                          landmarks: Optional[Dict[str, List[float]]] = None,
                          include_debug: bool = False,
                          track_id: str = "default",
                          stable_time_sec: float = 1.0) -> Dict[str, Any]:
        """
        Detect attributes for a specific person in the image.
        
        Args:
            image: Full BGR image
            person_bbox: (x1, y1, x2, y2)
            landmarks: Optional dict of normalized landmark points (e.g. {'left_eye': [x,y], ...})
            include_debug: Whether to return region coordinates.
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
            return {}
            
        person_crop = image[y1:y2, x1:x2]
        
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Pass 1: Whole Body (Shirt, Outerwear)
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Region Mapping using Landmarks for Precision
        # These are used for visualization and specialized CLIP passes
        regions = {}
        h_img, w_img = image.shape[:2]
        
        def to_abs(norm_pt):
            return [int(norm_pt[0] * w_img), int(norm_pt[1] * h_img)]

        # 1. Accessories/Head (Default Heuristic)
        head_h = int((y2 - y1) * 0.35)
        regions["Accessory Pass"] = [x1, y1, x2, min(y1+head_h, h_img)]

        # 2. Refined Precision Regions (if landmarks exist)
        if landmarks:
            # Eyes (for Glasses)
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                le = to_abs(landmarks['left_eye'])
                re = to_abs(landmarks['right_eye'])
                eye_w = abs(re[0] - le[0]) * 1.5
                regions["Glasses"] = [int(min(le[0], re[0]) - eye_w*0.2), int(min(le[1], re[1]) - eye_w*0.3), 
                                      int(max(le[0], re[0]) + eye_w*0.2), int(max(le[1], re[1]) + eye_w*0.3)]
            
            # Ears (for Earrings)
            if 'left_ear' in landmarks:
                ear = to_abs(landmarks['left_ear'])
                regions["Earrings"] = [ear[0]-20, ear[1]-20, ear[0]+20, ear[1]+20]

            # Torso (for Shirt/Necklace)
            if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
                ls = to_abs(landmarks['left_shoulder'])
                rs = to_abs(landmarks['right_shoulder'])
                sh_w = abs(rs[0] - ls[0])
                regions["Clothing"] = [int(min(ls[0], rs[0]) - sh_w*0.2), int(min(ls[1], rs[1])), 
                                      int(max(ls[0], rs[0]) + sh_w*0.2), int(min(ls[1], rs[1]) + sh_w*1.5)]

        # specialized crops helper
        def get_crop_input(rect):
            cx1, cy1, cx2, cy2 = rect
            # Ensure within image bounds
            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(w_img, cx2), min(h_img, cy2)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0: return image_input # fallback
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            return self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        # Standard Pass (Whole Person)
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Features extraction
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Standard Head Pass (fallback/context)
            head_input = get_crop_input(regions["Accessory Pass"])
            head_features = self.model.encode_image(head_input)
            head_features /= head_features.norm(dim=-1, keepdim=True)
            
            # Precision Passes
            glasses_input = get_crop_input(regions["Glasses"]) if "Glasses" in regions else head_input
            earrings_input = get_crop_input(regions["Earrings"]) if "Earrings" in regions else head_input
            clothing_input = get_crop_input(regions["Clothing"]) if "Clothing" in regions else image_input

            # --- Competition Formatted Attributes ---
            competition_fmt = {}

            # 1. Gender 
            probs = self._get_probs(image_features, "Gender")
            competition_fmt["Gender >>"] = "Male" if probs[0] > probs[1] else "Female"
            
            # 2. Clothing color (Tops)
            clothing_feat = self.model.encode_image(clothing_input)
            clothing_feat /= clothing_feat.norm(dim=-1, keepdim=True)
            probs_color = self._get_probs(clothing_feat, "ShirtColor")
            color_idx = np.argmax(probs_color)
            color_prompt = self.attribute_prompts["ShirtColor"][color_idx]
            competition_fmt["Clothing color (Tops) >>"] = color_prompt.split("wearing a ")[1].split(" ")[0].capitalize()

            # 3. Hair color
            probs = self._get_probs(head_features, "HairColor")
            best_idx = np.argmax(probs)
            hair_prompt = self.attribute_prompts["HairColor"][best_idx]
            if "bald" in hair_prompt:
                 competition_fmt["Hair color >>"] = "Bald"
            else:
                 competition_fmt["Hair color >>"] = hair_prompt.split("with ")[1].split(" ")[0].capitalize()

            # 4. Age (mapping descriptive age to rough integer ±7)
            probs = self._get_probs(image_features, "Age")
            best_idx = np.argmax(probs)
            age_map = ["10", "16", "30", "60"] # Child, Teenager, Adult, Senior
            competition_fmt["Age >>"] = f"{age_map[best_idx]} \u00b17 years old"

            # 5. Wears Glasses
            glasses_feat = self.model.encode_image(glasses_input)
            glasses_feat /= glasses_feat.norm(dim=-1, keepdim=True)
            probs = self._get_probs(glasses_feat, "Eyewear")
            competition_fmt["Wears Glasses >>"] = "Yes" if probs[0] > 0.65 else "No"
            
            # 6. Wears Cap/Hat
            probs = self._get_probs(head_features, "Headwear")
            competition_fmt["Wears Cap/Hat >>"] = "Yes" if probs[0] > 0.4 else "No"

            # 7. Wears Mask
            probs = self._get_probs(head_features, "Mask")
            competition_fmt["Wears Mask >>"] = "Yes" if probs[0] > 0.6 else "No"
            
            # --- Temporal Smoothing (Hysteresis) ---
            now = time.time()
            if track_id not in self.history:
                self.history[track_id] = {}
                self.locked_attributes[track_id] = {}
                
            for key, val in competition_fmt.items():
                if key not in self.history[track_id]:
                    self.history[track_id][key] = []
                
                # Check required continuous duration based on the attribute
                required_duration = 1.5 if key == "Wears Glasses >>" else (stable_time_sec * 0.7)
                
                # Append current frame
                self.history[track_id][key].append((now, val))
                
                # Safely retain the last 10 seconds of history (plenty of time to accumulate the required duration)
                self.history[track_id][key] = [(t, v) for t, v in self.history[track_id][key] 
                                               if now - t <= 10.0]
                
                # Extract only the values that fall within our required time window to check stability
                valid_history = [(t, v) for t, v in self.history[track_id][key] if now - t <= required_duration]
                
                # If there are NO measurements, or just the current frame, we can't be sure it's stable over time
                if len(valid_history) >= 2:
                    history_vals = [v for t, v in valid_history]
                    elapsed_window = valid_history[-1][0] - valid_history[0][0]
                
                    # Use a robust majority vote instead of requiring 100% agreement
                    # Find the most common value in the recent window
                    majority_val = max(set(history_vals), key=history_vals.count)
                    majority_ratio = history_vals.count(majority_val) / len(history_vals)
                    
                    # Only lock if the dominant value makes up at least 60% of the observations
                    # AND the observation span is long enough
                    if majority_ratio >= 0.60 and elapsed_window >= (required_duration * 0.8):
                        if key == "Wears Glasses >>":
                            # PERMANENT YES LOCK: Only lock 'Yes'. Once it's 'Yes', it stays 'Yes' forever.
                            if majority_val == "Yes":
                                self.locked_attributes[track_id][key] = "Yes"
                        else:
                            # Standard smoothing logic for all other attributes (can transition back and forth)
                            self.locked_attributes[track_id][key] = majority_val
                    
                # Use the locked value if it exists; otherwise use current prediction
                competition_fmt[key] = self.locked_attributes[track_id].get(key, val)

            detected_attributes = competition_fmt

            if include_debug:
                active_regions = {}
                # Only show regions corresponding to DETECTED attributes
                for attr in detected_attributes:
                    if "Glasses" in attr and "Glasses" in regions:
                        active_regions["Glasses"] = regions["Glasses"]
                    elif "Earrings" in attr and "Earrings" in regions:
                        active_regions["Earrings"] = regions["Earrings"]
                    elif ("T-shirt" in attr or "Sweater" in attr) and "Clothing" in regions:
                        active_regions["Clothing"] = regions["Clothing"]
                    elif any(ha in attr for ha in ["Hat", "Hair"]):
                         active_regions["Head/Hair"] = regions["Accessory Pass"]
                
                # Fallback: if nothing else, show the passes
                if not active_regions:
                     active_regions = regions
                
                detected_attributes["_debug"] = {
                    "active_regions": active_regions
                }

        return detected_attributes

    def _get_probs(self, image_features, category: str) -> np.ndarray:
        """Calculate softmax probabilities for a category."""
        prompts, text_embeds = self.text_features[category]
        
        # similarity = (100.0 * image @ text.T).softmax(dim=-1)
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_embeds.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        return probs
