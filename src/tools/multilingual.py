from typing import Tuple, Optional
from google.cloud import translate_v2 as translate
from langdetect import detect
import logging
import re
from pathlib import Path


logger = logging.getLogger(__name__)

def detect_and_translate(text: str) -> Tuple[str, Optional[str], Optional[float]]:
    """Detect language and translate if needed"""
    try:
        # Detect language
        detected_lang = detect(text)
        
        if detected_lang == 'de':
            # Initialize translator
            client = translate.Client()
            
            # Translate text
            result = client.translate(
                text,
                target_language='en',
                source_language='de'
            )
            
            return text, result['translatedText'], 0.95
            
        return text, None, None
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text, None, None

def preprocess_financial_text(text: str) -> str:
    """Clean and format financial document text"""
    # Protect financial identifiers
    protected_patterns = [
        (r'(IBAN\s*[A-Z0-9\s]+)', r' \1 '),
        (r'(BIC\s*[A-Z0-9]+)', r' \1 '),
        (r'(VAT\s*[A-Z0-9]+)', r' \1 '),
        (r'(\d{2}/\d{3}/\d{5})', r' \1 '),
    ]
    
    # Clean formatting artifacts
    cleanup_patterns = [
        (r'[_—]{2,}', ' '),
        (r'\s+', ' '),
    ]
    
    # Structure preservation
    structure_patterns = [
        (r'(\d+[.,]\d{2})', r' \1 '),
        (r'([A-Z]{2}\d{2}.*)', r' \1 ')
    ]
    
    # Apply all patterns
    for pattern, replacement in (protected_patterns + cleanup_patterns + structure_patterns):
        text = re.sub(pattern, replacement, text)
    
    return text.strip()

def save_multilingual_output(
    original_text: str,
    translated_text: Optional[str],
    output_path: Path
) -> None:
    """Save original and translated text to file"""
    output_content = f"Original Text:\n{original_text}\n"
    if translated_text:
        output_content += f"\nTranslated Text:\n{translated_text}"
    
    output_path.write_text(output_content) 