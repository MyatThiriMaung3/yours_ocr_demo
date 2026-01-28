# Yours OCR - Handwritten Text Recognition System

A web-based Optical Character Recognition (OCR) system for detecting and recognizing handwritten English text from images.

## Developed For the Undergraduate IT project of TDTU
**Myat Thiri Maung**
Faculty of Information Technology  
Ton Duc Thang University  

## Demo Youtube Video
- [Yours OCR demo](https://youtu.be/gICOrlWt0oA?si=rYLlpNBVIGiDIrYc)

## Project Overview

This project implements a complete handwritten OCR web application that converts handwritten English text images into editable digital text. The system uses a two-model pipeline combining word detection and word recognition with post-processing spelling correction.

### Key Features
- ✅ Single and multiple image upload support
- ✅ Automatic word detection in handwritten documents
- ✅ Handwritten text recognition using CNN-BiLSTM architecture
- ✅ Spelling correction with SymSpell algorithm
- ✅ Editable text interface
- ✅ Copy to clipboard functionality
- ✅ Download results as .txt files
- ✅ Responsive React-based UI

## System Architecture

### Two-Model Pipeline
1. **Word Detection Model**: Identifies and localizes handwritten word regions
2. **Word Recognition Model**: Recognizes text from cropped word images

### Technology Stack
- **Frontend**: React.js
- **Backend**: FastAPI (Python)
- **ML Framework**: TensorFlow/Keras
- **Spelling Correction**: SymSpell
- **Dataset**: IAM Handwriting Database

## Models

### Word Detection Model
- **Source**: [WordDetectorNN](https://github.com/githubharald/WordDetectorNN)
- **Type**: Dense prediction model for pixel-wise word detection
- **Approach**: Combines semantic segmentation with geometric regression
- **Dataset**: IAM Handwriting Database (~1,500 pages, ~115,000 words)

### Word Recognition Model
- **Architecture**: CNN-BiLSTM with CTC (Connectionist Temporal Classification)
- **Base Model**: Improved version of [handwritten-model](https://github.com/ketoin23/Handwritten-Text-Recognition)
- **Parameters**: 1.67M (increased from 424K in base model)
- **Dataset**: IAM_Words (96,456 cropped word images)

#### Model Improvements
| Aspect | Base Model | Improved Model |
|--------|------------|----------------|
| Parameters | ~424K | ~1.67M |
| Batch Normalization | ❌ | ✅ |
| Data Augmentation | Limited | Enhanced |
| Learning Rate | Fixed | Scheduled |
| Character Accuracy | ~74% | **83.65%** |
| Word Accuracy | ~63% | **71.45%** |

#### Data Split
| Split | Percentage | Images |
|-------|------------|--------|
| Training | 90% | 86,810 |
| Validation | 5% | 4,823 |
| Testing | 5% | 4,823 |
| **Total** | **100%** | **96,456** |

## Performance

### Word Recognition Model Metrics
- **Character-level Accuracy**: 83.65%
- **Word-level Accuracy**: 71.45%
- **Word-level Error Rate**: 28.65%
- **Training Epochs**: 11 (with early stopping)

### Model Features
- Distortion-free image resizing with padding
- Random brightness and contrast augmentation
- Batch normalization for stable training
- CTC decoding for variable-length sequences

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Clone the repository
```bash
git clone https://github.com/MyatThiriMaung3/yours_ocr_demo.git
cd yours_ocr_demo
```

2. Navigate to backend directory
```bash
cd backend
```

3. Install Python dependencies
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server
```bash
uvicorn main:app --reload
```

5. Verify backend health
```bash
# Open browser and visit:
http://localhost:8000/health
```

### Frontend Setup

1. Open new terminal and navigate to frontend directory
```bash
cd frontend/yours_ocr_react
```

2. Install dependencies
```bash
npm install
```

3. Start development server
```bash
npm run dev
```

4. Access the application
```bash
# Open browser and visit:
http://localhost:5173
```

## API Endpoints

### Health Check
```
GET /health
```
Returns backend status and model availability.

### Extract Text (OCR)
```
POST /extract-text
Content-Type: multipart/form-data

Parameters:
- file: Image file (PNG, JPG, JPEG)

Response:
{
  "full_text": "recognized text...",
  "words": [...],
  "image_name": "example.png"
}
```

### Spelling Correction
```
POST /check-spelling
Content-Type: application/json

Body:
{
  "text": "text to correct"
}

Response:
{
  "corrected_text": "corrected text..."
}
```

### Spelling Correction Rules
The system excludes certain words from spell-checking to preserve accuracy:
- ✅ Uppercase words (e.g., "OCR", "CNN")
- ✅ Words with initial capital letters (e.g., proper nouns)
- ✅ Words shorter than 3 characters

## Key Achievements

- Successfully implemented end-to-end handwritten OCR pipeline
- Achieved **9.65% improvement** in character-level accuracy over base model
- Achieved **8.45% improvement** in word-level accuracy over base model
- Integrated word detection and recognition into functional web application
- Implemented smart spelling correction with filtering rules
- Created user-friendly interface with multiple export options

## References

### Datasets
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [IAM_Words Subset](https://git.io/J0fjL)

### Models & Code
- [WordDetectorNN](https://github.com/githubharald/WordDetectorNN) - Word Detection Model
- [Handwritten Text Recognition](https://github.com/ketoin23/Handwritten-Text-Recognition) - Base Recognition Model

### Technologies
- [Python Documentation](https://docs.python.org/3/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [SymSpellPy](https://github.com/mammothb/symspellpy)

### Design
- [Figma Design](https://www.figma.com/) - UI/UX Design

## License

This project is part of an undergraduate IT project at Ton Duc Thang University.

## Contact

**Myat Thiri Maung**
Ton Duc Thang University  
Faculty of Information Technology

---

*Project completed: January 2026*
