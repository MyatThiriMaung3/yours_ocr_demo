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

Copy your trained models to respective paths.
```
backend/trained_models/detector/
backend/trained_models/recognizer/

```

### 4. Run Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```
GET http://localhost:8000/health
```

### Extract Text
```
POST http://localhost:8000/extract-text
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

### Check spelling with symspell

```
http://localhost:8000/check-spelling
```
Content-Type:
```
json
{
  "text": "John works at NASA and recieved the documnet"
}
```

Response:
```
json
{
    "success": true,
    "corrected_text": "John works at NASA and received the document"
}
```
