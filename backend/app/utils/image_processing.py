import cv2
import numpy as np
from PIL import Image
import io


def validate_image(file_bytes: bytes) -> bool:
    """Validate if uploaded file is a valid image"""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return True
    except:
        return False


def save_temp_image(file_bytes: bytes, filename: str) -> str:
    """Save uploaded image temporarily"""
    import tempfile
    import os
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    
    with open(temp_path, 'wb') as f:
        f.write(file_bytes)
    
    return temp_path