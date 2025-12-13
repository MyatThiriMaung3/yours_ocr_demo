from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
from typing import Dict
import traceback
import logging
from datetime import datetime

from app.models.ocr_pipeline import OCRPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR API", version="2.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
DETECTOR_PATH = BASE_DIR / "trained_models" / "detector" / "weights"
DETECTOR_METADATA_PATH = BASE_DIR / "trained_models" / "detector" / "metadata.json"
RECOGNIZER_PATH = BASE_DIR / "trained_models" / "recognizer" / "HTR_prediction_model.keras"
CHAR_CONFIG_PATH = BASE_DIR / "trained_models" / "recognizer" / "char_config.json"
TEMP_DIR = BASE_DIR / "temp"
CROPPED_WORDS_DIR = BASE_DIR / "cropped_words"  # New directory for cropped words

# Create directories
TEMP_DIR.mkdir(exist_ok=True)
CROPPED_WORDS_DIR.mkdir(exist_ok=True)

# Initialize pipeline
ocr_pipeline = None
initialization_error = None

try:
    logger.info(f"Attempting to load models...")
    logger.info(f"Detector path: {DETECTOR_PATH}")
    logger.info(f"Detector metadata: {DETECTOR_METADATA_PATH}")
    logger.info(f"Recognizer path: {RECOGNIZER_PATH}")
    logger.info(f"Char config path: {CHAR_CONFIG_PATH}")
    
    # Check if all paths exist
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(f"Detector weights not found: {DETECTOR_PATH}")
    if not DETECTOR_METADATA_PATH.exists():
        raise FileNotFoundError(f"Detector metadata not found: {DETECTOR_METADATA_PATH}")
    if not RECOGNIZER_PATH.exists():
        raise FileNotFoundError(f"Recognizer model not found: {RECOGNIZER_PATH}")
    if not CHAR_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Char config not found: {CHAR_CONFIG_PATH}")
    
    ocr_pipeline = OCRPipeline(
        detector_path=str(DETECTOR_PATH),
        recognizer_path=str(RECOGNIZER_PATH),
        char_config_path=str(CHAR_CONFIG_PATH),
        detector_metadata_path=str(DETECTOR_METADATA_PATH)
    )
    logger.info("OCR Pipeline initialized successfully!")
except Exception as e:
    initialization_error = str(e)
    logger.error(f"Failed to initialize OCR Pipeline: {e}")
    logger.error(traceback.format_exc())

@app.get("/")
async def root():
    return {
        "message": "OCR API with CTC-based Recognition",
        "version": "2.0",
        "status": "running",
        "pipeline_loaded": ocr_pipeline is not None,
        "initialization_error": initialization_error
    }

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy" if ocr_pipeline is not None else "degraded",
        "model_loaded": ocr_pipeline is not None,
        "model_type": "CTC-based Handwritten Text Recognition",
        "initialization_error": initialization_error,
        "paths": {
            "detector": str(DETECTOR_PATH.exists()),
            "detector_metadata": str(DETECTOR_METADATA_PATH.exists()),
            "recognizer": str(RECOGNIZER_PATH.exists()),
            "char_config": str(CHAR_CONFIG_PATH.exists())
        }
    }

@app.post("/api/v1/extract-text")
async def extract_text(file: UploadFile = File(...)) -> Dict:
    """
    Extract text from uploaded image using OCR
    """
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    if ocr_pipeline is None:
        logger.error("OCR pipeline not initialized")
        raise HTTPException(
            status_code=500, 
            detail=f"OCR pipeline not initialized. Error: {initialization_error}"
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_filepath = TEMP_DIR / temp_filename
    
    # Create a folder name based on original filename and timestamp
    original_name = Path(file.filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cropped_folder_name = f"{original_name}_{timestamp}_words"
    cropped_output_dir = CROPPED_WORDS_DIR / cropped_folder_name
    
    try:
        # Save uploaded file
        logger.info(f"Saving file to: {temp_filepath}")
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info("Starting OCR processing...")
        logger.info(f"Cropped words will be saved to: {cropped_output_dir}")
        
        # Process image with cropped words saving enabled
        results = ocr_pipeline.process_image(
            str(temp_filepath), 
            padding=0,
            save_cropped_words=True,
            output_dir=str(cropped_output_dir)
        )
        
        logger.info(f"OCR completed. Found {len(results['words'])} words")
        logger.info(f"Cropped words saved to: {results.get('cropped_words_dir')}")
        
        return {
            "success": True,
            "filename": file.filename,
            "results": results,
            "cropped_words_folder": cropped_folder_name
        }
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )
    
    finally:
        # Clean up temp file
        if temp_filepath.exists():
            temp_filepath.unlink()
            logger.info(f"Cleaned up temp file: {temp_filepath}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
