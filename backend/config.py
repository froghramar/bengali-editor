"""
Configuration settings for the Bengali Text Editor Backend
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration: Switch between implementations using environment variables
# These can be set in a .env file (see .env.example) or as system environment variables
# Set USE_GEMINI_COMPLETE=true to use Gemini for completion
# Set USE_GEMINI_TRANSLITERATE=true to use Gemini for transliteration
# Authentication: Use either GEMINI_API_KEY (API key) or GOOGLE_APPLICATION_CREDENTIALS (service account JSON path)
USE_GEMINI_COMPLETE = os.getenv("USE_GEMINI_COMPLETE", "false").lower() == "true"
USE_GEMINI_TRANSLITERATE = os.getenv("USE_GEMINI_TRANSLITERATE", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
