import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json

from .net import WordDetectorNet
from .coding import decode, fg_by_cc
from .aabb import AABB
from .aabb_clustering import cluster_aabbs
from .utils import compute_scale_down


class WordDetector:
    """
    Word detector using exact logic from original infer.py
    """
    
    def __init__(self, model_path: str, metadata_path: str, device: str = "cpu"):
        """
        Initialize word detector
        
        Args:
            model_path: Path to detector weights file
            metadata_path: Path to metadata.json
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            print(f"Loaded detector - Epoch: {self.metadata.get('epoch')}, F1: {self.metadata.get('val_f1')}")
        
        # Load model
        self.model = WordDetectorNet()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.input_size = WordDetectorNet.input_size
        self.output_size = WordDetectorNet.output_size
        self.scale_down = compute_scale_down(self.input_size, self.output_size)
        self.max_side_len = 1024
    
    def _ceil32(self, val):
        """Round up to nearest multiple of 32"""
        if val % 32 == 0:
            return val
        return (val // 32 + 1) * 32
    
    def _preprocess_image(self, image: np.ndarray):
        """
        Preprocess image - EXACT logic from DataLoaderImgFile
        
        Args:
            image: BGR image from cv2.imread or numpy array
            
        Returns:
            tuple: (preprocessed_tensor, scale_factor, original_grayscale)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            orig = image.copy()
        
        # Calculate scale factor
        f = min(self.max_side_len / orig.shape[0], self.max_side_len / orig.shape[1])
        
        # Resize if too large
        if f < 1:
            orig = cv2.resize(orig, dsize=None, fx=f, fy=f)
        
        # Pad to multiples of 32
        img = np.ones((self._ceil32(orig.shape[0]), self._ceil32(orig.shape[1])), np.uint8) * 255
        img[:orig.shape[0], :orig.shape[1]] = orig
        
        # Normalize to [-0.5, 0.5]
        img = (img / 255 - 0.5).astype(np.float32)
        
        # Add batch and channel dimensions: [1, 1, H, W]
        imgs = img[None, None, ...]
        imgs = torch.from_numpy(imgs).to(self.device)
        
        return imgs, f, orig
    
    def detect_words(self, image: np.ndarray, max_aabbs: int = 1000, padding: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Detect word bounding boxes in image
        
        Args:
            image: Input image as numpy array (BGR format from cv2.imread)
            max_aabbs: Maximum number of bounding boxes to return
            padding: Pixels to add around each bounding box (default: 5)
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        orig_h, orig_w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        
        # Preprocess
        imgs, scale_factor, orig_gray = self._preprocess_image(image)
        
        # Inference with softmax (matching eval.py)
        with torch.no_grad():
            y = self.model(imgs, apply_softmax=True)
            y_np = y.to('cpu').numpy()
        
        # Decode predictions (matching eval.py)
        scale_up = 1 / self.scale_down
        pred_map = y_np[0]
        
        aabbs = decode(pred_map, comp_fg=fg_by_cc(0.5, max_aabbs), f=scale_up)
        
        # Clip to image bounds
        h, w = imgs.shape[2:]
        img_bound = AABB(0, w - 1, 0, h - 1)
        aabbs = [aabb.clip(img_bound) for aabb in aabbs]
        aabbs = [aabb for aabb in aabbs if aabb.area() > 0]
        
        # Cluster overlapping boxes (matching eval.py)
        clustered_aabbs = cluster_aabbs(aabbs)
        
        # Scale back to original image size WITH PADDING
        final_bboxes = []
        
        for aabb in clustered_aabbs:
            # Scale back: undo the resize we did in preprocessing
            scaled_aabb = aabb.scale(1/scale_factor, 1/scale_factor)
            
            # Add padding
            x1 = max(0, int(scaled_aabb.xmin) - padding)
            y1 = max(0, int(scaled_aabb.ymin) - padding)
            x2 = min(orig_w - 1, int(scaled_aabb.xmax) + padding)
            y2 = min(orig_h - 1, int(scaled_aabb.ymax) + padding)
            
            # Only add valid boxes
            if x2 > x1 and y2 > y1:
                final_bboxes.append((x1, y1, x2, y2))
        
        # keep original detection order which is already correct
        # The detector already outputs in reading order (top-to-bottom, left-to-right)
        
        return final_bboxes