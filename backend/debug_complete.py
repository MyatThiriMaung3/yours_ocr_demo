#!/usr/bin/env python3
"""
Complete OCR debugging tool - SIMPLIFIED
- Shows detection visualization
- Saves individual word crops
- Shows recognition results
- NO reading order changes (detector order is already correct!)
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))

from app.models.word_detector import WordDetector
from app.models.word_recognizer import WordRecognizer


def create_visualization(image, bboxes, words=None, output_path="debug_visualization.jpg"):
    """Create visualization with bounding boxes and recognized words"""
    img_vis = image.copy()
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box (green)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw order number (blue)
        cv2.putText(img_vis, str(i + 1), (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw recognized word below box (red) if provided
        if words and i < len(words):
            word = words[i][:20]  # Truncate long words
            cv2.putText(img_vis, word, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imwrite(output_path, img_vis)
    print(f"✓ Saved visualization: {output_path}")
    return img_vis


def save_word_crops(image, bboxes, output_dir="debug_words"):
    """Save individual word crops"""
    Path(output_dir).mkdir(exist_ok=True)
    
    word_images = []
    
    print(f"\n{'='*60}")
    print("WORD CROPS")
    print(f"{'='*60}\n")
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        word_img = image[y1:y2, x1:x2]
        h, w = word_img.shape[:2]
        
        # Save crop
        crop_path = f"{output_dir}/word_{i+1:03d}.jpg"
        cv2.imwrite(crop_path, word_img)
        
        is_valid = w >= 5 and h >= 5
        status = "✓" if is_valid else "✗"
        
        print(f"{status} Word {i+1:3d}: {w:3d}x{h:2d}px")
        
        # Convert to PIL for recognition
        if is_valid:
            word_img_rgb = cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB)
            word_images.append(Image.fromarray(word_img_rgb))
        else:
            word_images.append(None)
    
    print(f"\n✓ Saved {len(bboxes)} crops to: {output_dir}/")
    
    return word_images


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to test image')
    parser.add_argument('--padding', type=int, default=10, help='Bbox padding (default: 10)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--ground-truth', help='Ground truth text file (optional)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SIMPLIFIED OCR PIPELINE DEBUG")
    print("="*60 + "\n")
    
    # Load models
    print("[1] Loading models...")
    base_path = Path(__file__).parent / "trained_models"
    
    detector = WordDetector(
        model_path=str(base_path / "detector" / "weights"),
        metadata_path=str(base_path / "detector" / "metadata.json"),
        device=args.device
    )
    
    recognizer = WordRecognizer(
        model_path=str(base_path / "recognizer" / "word_model_2809_words.pth"),
        encoder_path=str(base_path / "recognizer" / "word_label_encoder_2809_words.pkl"),
        device=args.device
    )
    
    print("✓ Models loaded\n")
    
    # Read image
    print(f"[2] Reading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"✗ Failed to read image")
        return
    
    print(f"✓ Image shape: {image.shape}\n")
    
    # Detect words WITH PADDING
    print(f"[3] Detecting words (padding={args.padding}px)...")
    bboxes = detector.detect_words(image, padding=args.padding)
    print(f"✓ Detected {len(bboxes)} words\n")
    
    if len(bboxes) == 0:
        print("✗ No words detected!")
        return
    
    # Save detection visualization
    print("[4] Creating detection visualization...")
    create_visualization(image, bboxes, output_path="debug_detection.jpg")
    
    # Extract word crops
    print("\n[5] Extracting word crops...")
    word_images = save_word_crops(image, bboxes)
    
    # Recognize words
    print(f"\n[6] Recognizing words...")
    valid_images = [img for img in word_images if img is not None]
    
    if not valid_images:
        print("✗ No valid word crops!")
        return
    
    print(f"Recognizing {len(valid_images)} valid crops...")
    recognized = recognizer.recognize_words_batch(valid_images)
    
    # Map back to all positions
    words = []
    rec_idx = 0
    for img in word_images:
        if img is not None:
            words.append(recognized[rec_idx])
            rec_idx += 1
        else:
            words.append('[INVALID]')
    
    print(f"✓ Recognized {len(recognized)} words\n")
    
    # Print results
    print(f"{'='*60}")
    print("RECOGNITION RESULTS (ORIGINAL ORDER)")
    print(f"{'='*60}\n")
    
    for i, (bbox, word) in enumerate(zip(bboxes, words)):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        print(f"{i+1:3d}. '{word:20s}' | Box: ({x1:4d},{y1:3d})->({x2:4d},{y2:3d}) | {w:3d}x{h:2d}px")
    
    # Create full visualization with text
    print(f"\n[7] Creating final visualization...")
    create_visualization(image, bboxes, words, "debug_with_text.jpg")
    
    # Print full text
    print(f"\n{'='*60}")
    print("EXTRACTED TEXT (NO REORDERING)")
    print(f"{'='*60}\n")
    
    valid_words = [w for w in words if w != '[INVALID]']
    full_text = ' '.join(valid_words)
    print(full_text)
    print()
    
    # Compare with ground truth
    if args.ground_truth:
        print(f"{'='*60}")
        print("GROUND TRUTH COMPARISON")
        print(f"{'='*60}\n")
        
        try:
            with open(args.ground_truth, 'r', encoding='utf-8') as f:
                gt = f.read().strip()
            
            print(f"Ground Truth:\n{gt}\n")
            print(f"Recognized:\n{full_text}\n")
            
            gt_words = gt.split()
            print(f"Word count: GT={len(gt_words)}, Recognized={len(valid_words)}")
            
        except Exception as e:
            print(f"✗ Could not load ground truth: {e}")
    
    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    print("✓ debug_detection.jpg      - Detections with order numbers")
    print("✓ debug_with_text.jpg      - Detections + recognized text")
    print("✓ debug_words/             - Individual word crops")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()