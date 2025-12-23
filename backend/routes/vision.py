"""
API routes for vision analysis
"""
import logging
from fastapi import UploadFile, File, HTTPException
from schemas import VisionAnalysisRequest, VisionAnalysisResponse
from services.gemini import analyze_vision

logger = logging.getLogger(__name__)


async def analyze_file(
    file: UploadFile = File(...),
    prompt: str = ""
) -> VisionAnalysisResponse:
    """
    Analyze image or PDF file using Gemini Vision model
    
    Accepts image (PNG, JPEG, etc.) or PDF file and returns analysis.
    
    Args:
        file: Image or PDF file to analyze
        prompt: Optional prompt/context from the editor
        
    Returns:
        VisionAnalysisResponse with summary, html_output, and extracted_text
    """
    try:
        # Read file content
        file_content = await file.read()
        file_type = file.content_type or ''
        
        logger.info(f"Analyzing file: {file.filename}, type: {file_type}, size: {len(file_content)} bytes")
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Validate file type
        if not (file_type.startswith('image/') or file_type == 'application/pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_type}. Please upload an image or PDF file."
            )
        
        # Analyze with Gemini Vision
        result = await analyze_vision(file_content, file_type, prompt)
        
        logger.info(f"Vision analysis completed for {file.filename}")
        
        return VisionAnalysisResponse(
            summary=result["summary"],
            html_output=result["html_output"],
            extracted_text=result.get("extracted_text", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in vision analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Vision analysis error: {str(e)}")
