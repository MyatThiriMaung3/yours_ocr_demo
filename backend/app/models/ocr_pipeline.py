import cv2
import numpy as np
from PIL import Image
from typing import List, Dict
from .word_detector import WordDetector
from .word_recognizer import WordRecognizer


class OCRPipeline:
    def __init__(
        self,
        detector_weights: str,
        detector_metadata: str,
        recognizer_model: str,
        recognizer_encoder: str,
        device: str = "cpu"
    ):
        """Complete OCR pipeline"""
        print("Initializing OCR Pipeline...")
        self.detector = WordDetector(detector_weights, detector_metadata, device)
        self.recognizer = WordRecognizer(recognizer_model, recognizer_encoder, device)
        print("OCR Pipeline ready!")
    
    def process_image(self, image_path: str, padding: int = 10) -> Dict:
        """
        Process image and extract text
        
        Args:
            image_path: Path to input image
            padding: Padding around bounding boxes (default: 10px)
            
        Returns:
            Dictionary with detected words and their bounding boxes
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"Processing image: {image.shape}")
        
        # Detect word bounding boxes WITH PADDING
        bboxes = self.detector.detect_words(image, padding=padding)
        print(f"Detected {len(bboxes)} words")
        
        # Extract word images and recognize
        word_images = []
        valid_bboxes = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            word_img = image[y1:y2, x1:x2]
            
            # Skip if bbox is too small
            if word_img.shape[0] < 5 or word_img.shape[1] < 5:
                print(f"  Skipping small crop: {word_img.shape}")
                continue
            
            # Convert BGR to RGB for PIL
            word_img_rgb = cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB)
            word_images.append(Image.fromarray(word_img_rgb))
            valid_bboxes.append(bbox)
        
        # Recognize words (batch processing)
        if word_images:
            words = self.recognizer.recognize_words_batch(word_images)
        else:
            words = []
        
        print(f"Recognized {len(words)} words")
        
        # Combine results - NO REORDERING!
        # The detector already gives us correct reading order
        results = {
            "words": [],
            "full_text": " ".join(words)
        }
        
        for i, (bbox, word) in enumerate(zip(valid_bboxes, words)):
            results["words"].append({
                "text": word,
                "bbox": {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3])
                },
                "confidence": 1.0,
                "order": i + 1  # Keep original order
            })
        
        return results