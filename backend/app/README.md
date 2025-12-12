# OCR API Backend

Word Detection and Recognition API using FastAPI.

## Features

- Word detection using custom trained model
- Word recognition using ResNet18 (2809 words vocabulary)
- REST API with FastAPI
- CORS enabled for frontend integration

## Setup

### 1. Create Virtual Environment
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Place Model Files

Copy your trained models to:
```
backend/trained_models/
├── detector/
│   ├── weights
│   └── metadata.json
└── recognizer/
    ├── word_model_2809_words.pth
    └── word_label_encoder_2809_words.pkl
```

### 4. Run Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```
GET http://localhost:8000/api/v1/health
```

### Extract Text
```
POST http://localhost:8000/api/v1/extract-text
Content-Type: multipart/form-data

file: <image file>
```

Response:
```json
{
  "success": true,
  "filename": "example.jpg",
  "results": {
    "words": [
      {
        "text": "hello",
        "bbox": {
          "x1": 10,
          "y1": 20,
          "x2": 100,
          "y2": 50
        },
        "confidence": 1.0
      }
    ],
    "full_text": "hello world"
  }
}
```

## Testing

### Using cURL
```bash
curl -X POST "http://localhost:8000/api/v1/extract-text" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Using Browser
Navigate to: http://localhost:8000/docs

## Directory Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── aabb.py
│   │   ├── aabb_clustering.py
│   │   ├── coding.py
│   │   ├── net.py
│   │   ├── resnet.py
│   │   ├── utils.py
│   │   ├── word_detector.py
│   │   ├── word_recognizer.py
│   │   └── ocr_pipeline.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── image_processing.py
│   └── api/
│       ├── __init__.py
│       └── routes.py
├── trained_models/
│   ├── detector/
│   └── recognizer/
├── requirements.txt
├── .env
└── README.md
```