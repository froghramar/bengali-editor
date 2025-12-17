"""
Bengali Text Auto-completion Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from typing import List
import logging
import re

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

# Global model variables
model = None
tokenizer = None
device = None

# Transliteration model variables
transliteration_model = None
transliteration_tokenizer = None

class CompletionRequest(BaseModel):
    text: str
    max_suggestions: int = 5
    max_length: int = 20  # Max tokens to generate

class CompletionResponse(BaseModel):
    suggestions: List[str]
    input_text: str

def normalize_bengali_text(text):
    """
    Basic Bengali text normalization
    For full normalization, install: pip install git+https://github.com/csebuetnlp/normalizer
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer, device
    global transliteration_model, transliteration_tokenizer
    
    try:
        logger.info("Loading model...")
        
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        model_name = "bigscience/bloom-560m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model vocabulary size: {len(tokenizer)}")


        # Load Banglish to Bengali transliteration model
        logger.info("Loading transliteration model...")
        transliteration_model_name = "Mdkaif2782/banglish-to-bangla"
        
        try:
            transliteration_tokenizer = MBart50TokenizerFast.from_pretrained(transliteration_model_name)
            transliteration_model = MBartForConditionalGeneration.from_pretrained(transliteration_model_name)
            transliteration_model.to(device)
            transliteration_model.eval()
            logger.info("Transliteration model loaded successfully!")
            logger.info(f"Transliteration model vocabulary size: {len(transliteration_tokenizer)}")
        except Exception as e:
            logger.warning(f"Could not load transliteration model: {e}")
            logger.warning("Transliteration endpoint will not be available")
            transliteration_model = None
            transliteration_tokenizer = None
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "device": str(device)
    }

@app.post("/complete", response_model=CompletionResponse)
async def get_completions(request: CompletionRequest):
    """
    Generate auto-completion suggestions for Bengali text using T5
    
    T5 works best with task prefixes. For completion, we can use:
    - "সম্পূর্ণ করুন: " (complete this)
    - Or use masking approach
    
    Args:
        request: CompletionRequest with text and parameters
        
    Returns:
        CompletionResponse with suggestions
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_text = normalize_bengali_text(request.text.strip())
        
        if not input_text:
            return CompletionResponse(suggestions=[], input_text=input_text)
        
        # T5 approach 1: Use task prefix for completion
        task_input = f"Complete: {input_text}"
        
        # Tokenize input
        input_ids = tokenizer.encode(
            task_input, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        # Generate completions using beam search
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=request.max_length + input_ids.shape[1],
                num_beams=request.max_suggestions * 2,
                num_return_sequences=request.max_suggestions,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                early_stopping=True,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
                length_penalty=1.0
            )
        
        # Decode and extract suggestions
        suggestions = []
        for output in outputs:
            # Decode the generated text
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            
            # Clean up the suggestion
            suggestion = generated_text.strip()
            
            # Remove the input text if it's repeated
            if suggestion.startswith(task_input):
               suggestion = suggestion.removeprefix(task_input).strip()

            # suggestions.append(suggestion)
            
            # Take first few words as suggestion (not the entire generation)
            words = suggestion.split()
            if words:
                # Suggest 1-10 words depending on context
                word_count = min(10, len(words))
                suggestion = ' '.join(words[:word_count])
                
                if suggestion and suggestion not in suggestions:
                    suggestions.append(suggestion)
        
        # Remove duplicates and empty suggestions
        suggestions = [s for s in suggestions if s][:request.max_suggestions]
        
        logger.info(f"Generated {len(suggestions)} suggestions for: '{input_text[:50]}...'")
        
        return CompletionResponse(
            suggestions=suggestions,
            input_text=input_text
        )
        
    except Exception as e:
        logger.error(f"Error generating completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/transliterate", response_model=CompletionResponse)
async def transliterate_text(request: CompletionRequest):
    """
    Transliterate Banglish (Roman/English letters) to Bengali
    Works like Avro but uses AI model instead of dictionary
    
    Examples:
    - "ami" -> "আমি"
    - "tumi kemon acho" -> "তুমি কেমন আছো"
    - "bangla" -> "বাংলা"
    
    Supports both:
    - Partial words: "am" -> ["আম", "আমি", "আমার"]
    - Complete phrases: "ami tomake bhalobashi" -> "আমি তোমাকে ভালোবাসি"
    """
    if transliteration_model is None or transliteration_tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Transliteration model not loaded. Please check backend logs."
        )
    
    try:
        input_text = request.text.strip()
        
        if not input_text:
            return CompletionResponse(suggestions=[], input_text=input_text)
        
        # Check if input is already in Bengali
        if any('\u0980' <= char <= '\u09FF' for char in input_text):
            # Input contains Bengali characters, return as-is
            return CompletionResponse(suggestions=[input_text], input_text=input_text)
        
        # Prepare input for mBART model
        inputs = transliteration_tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        # Generate multiple suggestions
        with torch.no_grad():
            outputs = transliteration_model.generate(
                **inputs,
                decoder_start_token_id=transliteration_tokenizer.lang_code_to_id["bn_IN"],
                max_length=128,
                num_beams=request.max_suggestions * 2,
                num_return_sequences=min(request.max_suggestions, 5),
                temperature=0.7,
                do_sample=True,
                top_k=50,
                early_stopping=True
            )
        
        # Decode suggestions
        suggestions = []
        for output in outputs:
            translated = transliteration_tokenizer.decode(output, skip_special_tokens=True)
            if translated and translated not in suggestions:
                suggestions.append(translated)
        
        # Remove duplicates while preserving order
        suggestions = list(dict.fromkeys(suggestions))[:request.max_suggestions]
        
        logger.info(f"Transliterated '{input_text}' -> {suggestions}")
        
        return CompletionResponse(
            suggestions=suggestions if suggestions else [input_text],
            input_text=input_text
        )
        
    except Exception as e:
        logger.error(f"Error in transliteration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)