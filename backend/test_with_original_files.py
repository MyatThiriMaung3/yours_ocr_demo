#!/usr/bin/env python3
"""Test detector using original files"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.models.word_detector import WordDetector


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to test image')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Testing with ORIGINAL files")
    print("="*60)
    
    # Initialize detector
    base_path = Path(__file__).parent / "trained_models"
    
    print("\nLoading model...")
    detector = WordDetector(
        model_path=str(base_path / "detector" / "weights"),
        metadata_path=str(base_path / "detector" / "metadata.json"),
        device=args.device
    )
    
    # Read image
    print(f"\nReading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"✗ Failed to read image")
        return
    
    print(f"✓ Image shape: {image.shape}")
    
    # Detect words
    print("\nDetecting words...")
    bboxes = detector.detect_words(image, max_aabbs=1000)
    
    print(f"\n{'='*60}")
    print(f"DETECTED {len(bboxes)} WORDS")
    print(f"{'='*60}\n")
    
    # Print and visualize
    img_vis = image.copy()
    
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Draw bbox
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_vis, str(i+1), (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        w, h = x2 - x1, y2 - y1
        print(f"  {i+1:3d}. ({x1:4d}, {y1:3d}) -> ({x2:4d}, {y2:3d}) | Size: {w:3d}x{h:2d}px")
    
    # Save visualization
    output_path = "test_with_original_files.jpg"
    cv2.imwrite(output_path, img_vis)
    
    print(f"\n{'='*60}")
    print(f"✓ Saved visualization: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()