"""
Bengali Text Auto-completion Backend - Main Application
"""
import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from models.loader import load_models, get_models
from routes.complete import get_completions
from routes.transliterate import transliterate_text
from routes.speech import speech_to_text as speech_to_text_handler
from routes.vision import analyze_file as analyze_file_handler
from schemas import CompletionRequest, CompletionResponse, SpeechToTextResponse, VisionAnalysisResponse
from config import USE_GEMINI_COMPLETE, USE_GEMINI_TRANSLITERATE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bengali Auto-completion API")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    await load_models()


@app.get("/")
async def root():
    """Health check endpoint"""
    models = get_models()
    return {
        "status": "running",
        "device": str(models['device']) if models['device'] else "N/A",
        "config": {
            "completion_backend": "Gemini Flash" if USE_GEMINI_COMPLETE else "Transformers",
            "transliteration_backend": "Gemini Flash" if USE_GEMINI_TRANSLITERATE else "Transformers",
            "gemini_configured": USE_GEMINI_COMPLETE or USE_GEMINI_TRANSLITERATE
        }
    }


@app.post("/complete", response_model=CompletionResponse)
async def complete_endpoint(request: CompletionRequest):
    """Generate auto-completion suggestions for Bengali text"""
    return await get_completions(request)


@app.post("/transliterate", response_model=CompletionResponse)
async def transliterate_endpoint(request: CompletionRequest):
    """Transliterate Banglish to Bengali and provide autocompletion suggestions"""
    return await transliterate_text(request)


@app.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text_endpoint(audio: UploadFile = File(...)):
    """Convert speech audio to text using Google Cloud Speech-to-Text API"""
    return await speech_to_text_handler(audio)


@app.post("/analyze-vision", response_model=VisionAnalysisResponse)
async def analyze_vision_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form("")
):
    """Analyze image or PDF file using Gemini Vision model"""
    return await analyze_file_handler(file, prompt)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
