from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from typing import Dict
from ..models.ocr_pipeline import OCRPipeline
from ..utils.image_processing import validate_image, save_temp_image

router = APIRouter()

# Global OCR pipeline instance
ocr_pipeline: OCRPipeline = None


def set_ocr_pipeline(pipeline: OCRPipeline):
    """Set the OCR pipeline instance"""
    global ocr_pipeline
    ocr_pipeline = pipeline


@router.post("/extract-text")
async def extract_text(file: UploadFile = File(...)) -> Dict:
    """
    Extract text from uploaded image
    
    Args:
        file: Uploaded image file (PNG, JPG, JPEG)
        
    Returns:
        JSON with extracted text and word bounding boxes
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read file
        contents = await file.read()
        
        # Validate image
        if not validate_image(contents):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Save temporarily
        temp_path = save_temp_image(contents, file.filename)
        
        # Process with OCR pipeline
        results = ocr_pipeline.process_image(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "results": results
        }
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "OCR API is running",
        "detector": "loaded" if ocr_pipeline else "not loaded",
        "recognizer": "loaded" if ocr_pipeline else "not loaded"
    }