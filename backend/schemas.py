"""
Pydantic models/schemas for API requests and responses
"""
from pydantic import BaseModel
from typing import List


class CompletionRequest(BaseModel):
    text: str
    max_suggestions: int = 5
    max_length: int = 20  # Max tokens to generate


class CompletionResponse(BaseModel):
    suggestions: List[str]
    input_text: str


class SpeechToTextResponse(BaseModel):
    text: str
    confidence: float = 0.0


class VisionAnalysisRequest(BaseModel):
    prompt: str = ""  # Optional prompt/context from editor


class VisionAnalysisResponse(BaseModel):
    summary: str
    html_output: str
    extracted_text: str = ""
