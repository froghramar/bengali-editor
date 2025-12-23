"""
Utility functions for text processing
"""
import re


def normalize_bengali_text(text: str) -> str:
    """
    Basic Bengali text normalization
    For full normalization, install: pip install git+https://github.com/csebuetnlp/normalizer
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
