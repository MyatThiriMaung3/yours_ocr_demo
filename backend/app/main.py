from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import torch
from .api.routes import router, set_ocr_pipeline
from .models.ocr_pipeline import OCRPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Word Detection & Recognition OCR API",
    description="API for detecting and recognizing words in images",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174"
    ],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR pipeline on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    
    print("=" * 60)
    print("Starting OCR API Server...")
    print("=" * 60)
    
    # Paths to models
    base_path = Path(__file__).parent.parent / "trained_models"
    
    detector_weights = str(base_path / "detector" / "weights")
    detector_metadata = str(base_path / "detector" / "metadata.json")
    recognizer_model = str(base_path / "recognizer" / "word_model_2809_words.pth")
    recognizer_encoder = str(base_path / "recognizer" / "word_label_encoder_2809_words.pkl")
    
    # Verify files exist
    print("\nVerifying model files...")
    for path, name in [
        (detector_weights, "Detector weights"),
        (detector_metadata, "Detector metadata"),
        (recognizer_model, "Recognizer model"),
        (recognizer_encoder, "Recognizer encoder")
    ]:
        if Path(path).exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: NOT FOUND - {path}")
            raise FileNotFoundError(f"Model file not found: {path}")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Using device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # Initialize pipeline
    try:
        pipeline = OCRPipeline(
            detector_weights=detector_weights,
            detector_metadata=detector_metadata,
            recognizer_model=recognizer_model,
            recognizer_encoder=recognizer_encoder,
            device=device
        )
        
        # Set pipeline in routes
        set_ocr_pipeline(pipeline)
        
        print("\n" + "=" * 60)
        print("✓ OCR Pipeline initialized successfully!")
        print("=" * 60)
        print("\nAPI Documentation: http://localhost:8000/docs")
        print("Health Check: http://localhost:8000/api/v1/health")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Failed to initialize OCR pipeline: {str(e)}")
        raise


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["OCR"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Word Detection & Recognition OCR API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/health",
            "extract_text": "/api/v1/extract-text"
        }
    }