from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
import shutil
import uuid
from typing import Dict
import traceback
import logging
from datetime import datetime
from symspellpy import SymSpell, Verbosity

from app.models.ocr_pipeline import OCRPipeline

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR API", version="2.0")

# cors configuration
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

# initialize symspell
sym_spell = None
symspell_error = None

try:
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    # load dictionary, the default English dictionary
    # dictionary file is in the symspellpy package
    import pkg_resources

    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )

    if sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
        logger.info("SymSpell dictionary loaded successfully")
    else:
        raise Exception("Failed to load SymSpell dictionary")

except Exception as e:
    symspell_error = str(e)
    logger.error(f"Failed to initialize SymSpell: {e}")

# initialize ocr pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
DETECTOR_PATH = BASE_DIR / "trained_models" / "detector" / "weights"
DETECTOR_METADATA_PATH = BASE_DIR / "trained_models" / "detector" / "metadata.json"
RECOGNIZER_PATH = (
    BASE_DIR / "trained_models" / "recognizer" / "htr_improved_prediction.keras"
)
CHAR_CONFIG_PATH = BASE_DIR / "trained_models" / "recognizer" / "char_config.json"
TEMP_DIR = BASE_DIR / "temp"

# create directories
TEMP_DIR.mkdir(exist_ok=True)

# initialize pipeline
ocr_pipeline = None
initialization_error = None

try:
    logger.info(f"Attempting to load models...")
    logger.info(f"Detector path: {DETECTOR_PATH}")
    logger.info(f"Detector metadata: {DETECTOR_METADATA_PATH}")
    logger.info(f"Recognizer path: {RECOGNIZER_PATH}")
    logger.info(f"Char config path: {CHAR_CONFIG_PATH}")

    # check if all paths exist
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(f"Detector weights not found: {DETECTOR_PATH}")
    if not DETECTOR_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Detector metadata not found: {DETECTOR_METADATA_PATH}"
        )
    if not RECOGNIZER_PATH.exists():
        raise FileNotFoundError(f"Recognizer model not found: {RECOGNIZER_PATH}")
    if not CHAR_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Char config not found: {CHAR_CONFIG_PATH}")

    ocr_pipeline = OCRPipeline(
        detector_path=str(DETECTOR_PATH),
        recognizer_path=str(RECOGNIZER_PATH),
        char_config_path=str(CHAR_CONFIG_PATH),
        detector_metadata_path=str(DETECTOR_METADATA_PATH),
    )
    logger.info("OCR Pipeline initialized successfully!")
except Exception as e:
    initialization_error = str(e)
    logger.error(f"Failed to initialize OCR Pipeline: {e}")
    logger.error(traceback.format_exc())


# pydantic model for spelling check request
class SpellingCheckRequest(BaseModel):
    text: str


def correct_text_with_symspell(text: str) -> str:
    """
    Correct spelling errors in text using SymSpell with intelligent gating

    Gating rules (skip correction if):
    - Word is capitalized (e.g., "John", "Paris")
    - Word is ALL CAPS (e.g., "USA", "OCR")
    - Word length ≤ 3 (e.g., "it", "the", "a")

    Preserves formatting and only corrects words with edit distance <= 2
    """
    if not sym_spell:
        return text

    lines = text.split("\n")
    corrected_lines = []

    for line in lines:
        if not line.strip():
            corrected_lines.append(line)
            continue

        words = line.split()
        corrected_words = []

        for word in words:
            # preserve punctuation
            punctuation = ""
            clean_word = word

            # extract trailing punctuation
            while clean_word and not clean_word[-1].isalnum():
                punctuation = clean_word[-1] + punctuation
                clean_word = clean_word[:-1]

            # extract leading punctuation
            leading_punct = ""
            while clean_word and not clean_word[0].isalnum():
                leading_punct += clean_word[0]
                clean_word = clean_word[1:]

            if clean_word:
                # skip correction if word is capitalized (proper noun), word is ALL CAPS (acronym), word length ≤ 3 (common short words)
                should_skip = (
                    (
                        len(clean_word) > 1
                        and clean_word[0].isupper()
                        and clean_word[1:].islower()
                    )  # Capitalized
                    or clean_word.isupper()  # all caps
                    or len(clean_word) <= 3  # short words
                )

                if should_skip:
                    # keep original word
                    corrected_words.append(word)
                else:
                    # get correction suggestions
                    suggestions = sym_spell.lookup(
                        clean_word.lower(), Verbosity.CLOSEST, max_edit_distance=2
                    )

                    if suggestions:
                        # use the best suggestion
                        corrected_word = suggestions[0].term
                        corrected_words.append(
                            leading_punct + corrected_word + punctuation
                        )
                    else:
                        corrected_words.append(word)
            else:
                corrected_words.append(word)

        corrected_lines.append(" ".join(corrected_words))

    return "\n".join(corrected_lines)


@app.get("/")
async def root():
    return {
        "message": "OCR API with CTC-based Recognition and SymSpell Correction",
        "version": "2.0",
        "status": "running",
        "pipeline_loaded": ocr_pipeline is not None,
        "symspell_available": sym_spell is not None,
        "initialization_error": initialization_error,
        "symspell_error": symspell_error,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if ocr_pipeline is not None else "degraded",
        "model_loaded": ocr_pipeline is not None,
        "model_type": "CTC-based Handwritten Text Recognition",
        "spell_checker": (
            "SymSpell (edit distance ≤ 2)" if sym_spell else "not available"
        ),
        "symspell_available": sym_spell is not None,
        "initialization_error": initialization_error,
        "symspell_error": symspell_error,
        "paths": {
            "detector": str(DETECTOR_PATH.exists()),
            "detector_metadata": str(DETECTOR_METADATA_PATH.exists()),
            "recognizer": str(RECOGNIZER_PATH.exists()),
            "char_config": str(CHAR_CONFIG_PATH.exists()),
        },
    }


@app.post("/check-spelling")
async def check_spelling(request: SpellingCheckRequest) -> Dict:
    """
    Post-process OCR text using SymSpell to correct spelling errors
    Uses edit distance ≤ 2 for corrections
    """
    logger.info(
        f"Received spelling check request for text of length: {len(request.text)}"
    )

    if sym_spell is None:
        logger.error("SymSpell not initialized")
        raise HTTPException(
            status_code=500, detail=f"SymSpell not initialized. Error: {symspell_error}"
        )

    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        logger.info("Starting SymSpell correction...")

        corrected_text = correct_text_with_symspell(request.text)

        logger.info(
            f"SymSpell correction completed. Corrected text length: {len(corrected_text)}"
        )

        return {"success": True, "corrected_text": corrected_text}

    except Exception as e:
        logger.error(f"Error processing spelling check: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error processing spelling check: {str(e)}"
        )


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)) -> Dict:
    """
    Extract text from uploaded image using OCR
    """
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")

    if ocr_pipeline is None:
        logger.error("OCR pipeline not initialized")
        raise HTTPException(
            status_code=500,
            detail=f"OCR pipeline not initialized. Error: {initialization_error}",
        )

    # validate file type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")

    # generate unique filename
    file_extension = Path(file.filename).suffix
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_filepath = TEMP_DIR / temp_filename

    try:
        # Save uploaded file
        logger.info(f"Saving file to: {temp_filepath}")
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("Starting OCR processing...")

        # process image
        results = ocr_pipeline.process_image(
            str(temp_filepath),
            padding=0,
        )

        logger.info(f"OCR completed. Found {len(results['words'])} words")

        return {
            "success": True,
            "filename": file.filename,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    finally:
        # clean up temp file
        if temp_filepath.exists():
            temp_filepath.unlink()
            logger.info(f"Cleaned up temp file: {temp_filepath}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
