"""
Transformers service for text completion and transliteration
"""
import logging
from fastapi import HTTPException
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MBartForConditionalGeneration, MBart50TokenizerFast

logger = logging.getLogger(__name__)


async def complete_with_transformers(
    input_text: str, 
    max_suggestions: int, 
    max_length: int,
    model,
    tokenizer,
    device
) -> list[str]:
    """Generate completions using transformers model (original implementation)"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
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
            max_length=max_length + input_ids.shape[1],
            num_beams=max_suggestions * 2,
            num_return_sequences=max_suggestions,
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
        
        # Take first few words as suggestion (not the entire generation)
        words = suggestion.split()
        if words:
            # Suggest 1-10 words depending on context
            word_count = min(10, len(words))
            suggestion = ' '.join(words[:word_count])
            
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)
    
    # Remove duplicates and empty suggestions
    suggestions = [s for s in suggestions if s][:max_suggestions]
    return suggestions


async def transliterate_with_transformers(
    input_text: str, 
    max_suggestions: int,
    transliteration_model,
    transliteration_tokenizer,
    device
) -> list[str]:
    """Transliterate Banglish to Bengali using transformers model (original implementation)"""
    if transliteration_model is None or transliteration_tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Transliteration model not loaded. Please check backend logs."
        )
    
    # Check if input is already in Bengali
    if any('\u0980' <= char <= '\u09FF' for char in input_text):
        # Input contains Bengali characters, return as-is
        return [input_text]
    
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
            num_beams=max_suggestions * 2,
            num_return_sequences=min(max_suggestions, 5),
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
    suggestions = list(dict.fromkeys(suggestions))[:max_suggestions]
    return suggestions if suggestions else [input_text]
