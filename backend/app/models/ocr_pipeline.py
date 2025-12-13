# import cv2
# import numpy as np
# from pathlib import Path
# from typing import List, Dict, Tuple
# from PIL import Image
# import os

# from .word_detector import WordDetector
# from .word_recognizer_ctc import WordRecognizerCTC

# class OCRPipeline:
#     def __init__(self, detector_path: str, recognizer_path: str, char_config_path: str, detector_metadata_path: str = None):
#         """
#         Initialize the OCR pipeline with detection and recognition models
        
#         Args:
#             detector_path: Path to word detection model weights
#             recognizer_path: Path to word recognition model (.keras or .h5)
#             char_config_path: Path to character configuration file
#             detector_metadata_path: Path to detector metadata.json (optional)
#         """
#         # If metadata path not provided, construct it from detector_path
#         if detector_metadata_path is None:
#             detector_dir = Path(detector_path).parent
#             detector_metadata_path = str(detector_dir / "metadata.json")
        
#         self.detector = WordDetector(detector_path, detector_metadata_path)
#         self.recognizer = WordRecognizerCTC(recognizer_path, char_config_path)
        
#     def process_image(self, image_path: str, padding: int = 10, save_cropped_words: bool = False, output_dir: str = None) -> Dict:
#         """
#         Process an image through the complete OCR pipeline
        
#         Args:
#             image_path: Path to the input image
#             padding: Padding around detected bounding boxes (pixels)
#             save_cropped_words: Whether to save cropped word images
#             output_dir: Directory to save cropped words (if save_cropped_words=True)
            
#         Returns:
#             Dictionary containing recognized words and full text
#         """
#         # Load image
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError(f"Failed to load image from {image_path}")
        
#         # Create output directory for cropped words if needed
#         cropped_words_dir = None
#         if save_cropped_words and output_dir:
#             cropped_words_dir = Path(output_dir)
#             cropped_words_dir.mkdir(parents=True, exist_ok=True)
        
#         # Detect words - pass the numpy array directly
#         bounding_boxes = self.detector.detect_words(image, max_aabbs=1000, padding=padding)
        
#         if not bounding_boxes:
#             return {
#                 "words": [],
#                 "full_text": "",
#                 "cropped_words_dir": str(cropped_words_dir) if cropped_words_dir else None
#             }
        
#         # Prepare word images for recognition
#         word_images = []
#         valid_boxes = []
        
#         for i, bbox in enumerate(bounding_boxes):
#             x1, y1, x2, y2 = bbox
            
#             # Crop word region (padding already added by detector)
#             word_img = image[y1:y2, x1:x2]
            
#             # Skip if crop is invalid
#             if word_img.size == 0 or word_img.shape[0] < 5 or word_img.shape[1] < 5:
#                 continue
            
#             # Save cropped word image if requested
#             if save_cropped_words and cropped_words_dir:
#                 word_filename = cropped_words_dir / f"word_{i:04d}.png"
#                 cv2.imwrite(str(word_filename), word_img)
            
#             # Convert to PIL Image for recognition
#             word_img_pil = Image.fromarray(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
#             word_images.append(word_img_pil)
#             valid_boxes.append((x1, y1, x2, y2, i))
        
#         # Recognize all words in batch
#         if word_images:
#             recognized_texts = self.recognizer.recognize_batch(word_images)
#         else:
#             recognized_texts = []
        
#         # Combine results and optionally rename saved images with recognized text
#         results = []
#         for idx, ((x1, y1, x2, y2, order), text) in enumerate(zip(valid_boxes, recognized_texts)):
#             results.append({
#                 "text": text,
#                 "bbox": {
#                     "x1": int(x1),
#                     "y1": int(y1),
#                     "x2": int(x2),
#                     "y2": int(y2)
#                 },
#                 "confidence": 1.0,  # CTC doesn't provide confidence scores directly
#                 "order": order
#             })
            
#             # Optionally rename saved image with recognized text
#             if save_cropped_words and cropped_words_dir:
#                 old_filename = cropped_words_dir / f"word_{order:04d}.png"
#                 # Sanitize text for filename (remove special characters)
#                 safe_text = "".join(c if c.isalnum() else "_" for c in text)
#                 safe_text = safe_text[:50]  # Limit length
#                 new_filename = cropped_words_dir / f"word_{order:04d}_{safe_text}.png"
#                 if old_filename.exists():
#                     old_filename.rename(new_filename)
        
#         # Sort by order (reading order from detector)
#         results.sort(key=lambda x: x["order"])
        
#         # Create full text
#         full_text = " ".join([word["text"] for word in results])
        
#         return {
#             "words": results,
#             "full_text": full_text,
#             "cropped_words_dir": str(cropped_words_dir) if cropped_words_dir else None
#         }


import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import os

from .word_detector import WordDetector
from .word_recognizer_ctc import WordRecognizerCTC

class OCRPipeline:
    def __init__(self, detector_path: str, recognizer_path: str, char_config_path: str, detector_metadata_path: str = None):
        """
        Initialize the OCR pipeline with detection and recognition models
        
        Args:
            detector_path: Path to word detection model weights
            recognizer_path: Path to word recognition model (.keras or .h5)
            char_config_path: Path to character configuration file
            detector_metadata_path: Path to detector metadata.json (optional)
        """
        # If metadata path not provided, construct it from detector_path
        if detector_metadata_path is None:
            detector_dir = Path(detector_path).parent
            detector_metadata_path = str(detector_dir / "metadata.json")
        
        self.detector = WordDetector(detector_path, detector_metadata_path)
        self.recognizer = WordRecognizerCTC(recognizer_path, char_config_path)
    
    def _sort_bboxes_reading_order(self, bboxes: List[Tuple[int, int, int, int]]) -> List[int]:
        """
        Sort bounding boxes in reading order (top-to-bottom, left-to-right)
        Uses improved line detection with Y-overlap checking
        
        Args:
            bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            List of indices representing the reading order
        """
        if not bboxes:
            return []
        
        # Convert to format with index and coordinates
        bbox_data = []
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            bbox_data.append({
                'index': i,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'height': y2 - y1,
                'width': x2 - x1
            })
        
        def boxes_on_same_line(box1, box2):
            """
            Check if two boxes are on the same line using Y-overlap
            Two boxes are on the same line if they have significant vertical overlap
            """
            # Calculate vertical overlap
            y_overlap = min(box1['y2'], box2['y2']) - max(box1['y1'], box2['y1'])
            
            # Get minimum height of the two boxes
            min_height = min(box1['height'], box2['height'])
            
            # Boxes are on same line if overlap is at least 50% of smaller box height
            if y_overlap >= min_height * 0.5:
                return True
            
            # Alternative check: if centers are very close vertically
            vertical_distance = abs(box1['center_y'] - box2['center_y'])
            avg_height = (box1['height'] + box2['height']) / 2
            
            if vertical_distance < avg_height * 0.4:
                return True
            
            return False
        
        # Group boxes into lines
        lines = []
        used = set()
        
        # Sort by Y position (top to bottom)
        sorted_boxes = sorted(bbox_data, key=lambda x: x['y1'])
        
        for current_box in sorted_boxes:
            if current_box['index'] in used:
                continue
            
            # Start new line with current box
            current_line = [current_box]
            used.add(current_box['index'])
            
            # Find all boxes that belong to the same line
            for candidate_box in sorted_boxes:
                if candidate_box['index'] in used:
                    continue
                
                # Check if candidate should be on the same line as any box in current_line
                for line_box in current_line:
                    if boxes_on_same_line(line_box, candidate_box):
                        current_line.append(candidate_box)
                        used.add(candidate_box['index'])
                        break
            
            lines.append(current_line)
        
        # Sort lines by their topmost Y position
        lines.sort(key=lambda line: min(box['y1'] for box in line))
        
        # Within each line, sort strictly by X position (left to right)
        sorted_indices = []
        for line in lines:
            # Sort by center_x for more stable sorting
            line.sort(key=lambda box: box['center_x'])
            sorted_indices.extend([box['index'] for box in line])
        
        return sorted_indices

    def process_image(self, image_path: str, padding: int = 10, save_cropped_words: bool = False, output_dir: str = None) -> Dict:
        """
        Process an image through the complete OCR pipeline
        
        Args:
            image_path: Path to the input image
            padding: Padding around detected bounding boxes (pixels)
            save_cropped_words: Whether to save cropped word images
            output_dir: Directory to save cropped words (if save_cropped_words=True)
            
        Returns:
            Dictionary containing recognized words and full text
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Create output directory for cropped words if needed
        cropped_words_dir = None
        if save_cropped_words and output_dir:
            cropped_words_dir = Path(output_dir)
            cropped_words_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect words - pass the numpy array directly
        bounding_boxes = self.detector.detect_words(image, max_aabbs=1000, padding=padding)
        
        if not bounding_boxes:
            return {
                "words": [],
                "full_text": "",
                "cropped_words_dir": str(cropped_words_dir) if cropped_words_dir else None
            }
        
        # Sort bounding boxes in reading order
        sorted_indices = self._sort_bboxes_reading_order(bounding_boxes)
        
        # Prepare word images for recognition (in sorted order)
        word_images = []
        valid_boxes = []
        
        for order, original_idx in enumerate(sorted_indices):
            bbox = bounding_boxes[original_idx]
            x1, y1, x2, y2 = bbox
            
            # Crop word region (padding already added by detector)
            word_img = image[y1:y2, x1:x2]
            
            # Skip if crop is invalid
            if word_img.size == 0 or word_img.shape[0] < 5 or word_img.shape[1] < 5:
                continue
            
            # Save cropped word image if requested
            if save_cropped_words and cropped_words_dir:
                word_filename = cropped_words_dir / f"word_{order:04d}.png"
                cv2.imwrite(str(word_filename), word_img)
            
            # Convert to PIL Image for recognition
            word_img_pil = Image.fromarray(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
            word_images.append(word_img_pil)
            valid_boxes.append((x1, y1, x2, y2, order))
        
        # Recognize all words in batch
        if word_images:
            recognized_texts = self.recognizer.recognize_batch(word_images)
        else:
            recognized_texts = []
        
        # Combine results and optionally rename saved images with recognized text
        results = []
        for idx, ((x1, y1, x2, y2, order), text) in enumerate(zip(valid_boxes, recognized_texts)):
            results.append({
                "text": text,
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "confidence": 1.0,  # CTC doesn't provide confidence scores directly
                "order": order
            })
            
            # Optionally rename saved image with recognized text
            if save_cropped_words and cropped_words_dir:
                old_filename = cropped_words_dir / f"word_{order:04d}.png"
                # Sanitize text for filename (remove special characters)
                safe_text = "".join(c if c.isalnum() else "_" for c in text)
                safe_text = safe_text[:50]  # Limit length
                new_filename = cropped_words_dir / f"word_{order:04d}_{safe_text}.png"
                if old_filename.exists():
                    old_filename.rename(new_filename)
        
        # Results are already in correct reading order
        # Create full text
        full_text = " ".join([word["text"] for word in results])
        
        return {
            "words": results,
            "full_text": full_text,
            "cropped_words_dir": str(cropped_words_dir) if cropped_words_dir else None
        }