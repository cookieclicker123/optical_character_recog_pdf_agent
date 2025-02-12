from pathlib import Path
import time
import uuid
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from transformers import DetrImageProcessor, DetrForObjectDetection
from langdetect import detect
from pdf2image import convert_from_path
import logging
from google.cloud import translate_v2 as translate
from dotenv import load_dotenv
import os
import re

from .data_model import OCRResult, ProcessingStatus, DocumentType, OcrFn

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GoogleTranslator:
    """Translation using Google Cloud Translate"""
    def __init__(self):
        self.client = translate.Client()
        
    def translate(self, text: str, source_lang: str = 'de') -> Tuple[str, float]:
        try:
            result = self.client.translate(
                text,
                target_language='en',
                source_language=source_lang
            )
            return result['translatedText'], 0.95
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return "", 0.0

def detect_document_type(filename: str) -> DocumentType:
    """Detect document type from filename"""
    filename = filename.lower()
    if "post" in filename:
        return DocumentType.POST
    elif "invoice" in filename:
        return DocumentType.INVOICE
    elif "receipt" in filename:
        return DocumentType.RECEIPT
    return DocumentType.OTHER

def process_text_with_translation(
    image: np.ndarray,
    doc_type: DocumentType,
    translator: GoogleTranslator
) -> Tuple[str, Optional[str], Optional[str], Optional[float]]:
    """Process text with language detection and translation if needed"""
    
    if doc_type == DocumentType.POST:
        # First OCR pass with German language data
        source_text = pytesseract.image_to_string(
            image,
            lang='deu',
            config='--psm 6'
        )
        
        try:
            # Clean and preprocess text before translation
            cleaned_text = preprocess_financial_document(source_text)
            logger.debug(f"Original text:\n{source_text}")
            logger.debug(f"Preprocessed text:\n{cleaned_text}")
            
            # Verify language
            detected_lang = detect(cleaned_text)
            if detected_lang == 'de':
                # Translate preprocessed text
                translated_text, translation_conf = translator.translate(cleaned_text)
                return source_text, detected_lang, translated_text, translation_conf
            else:
                # If not German, try English OCR
                english_text = pytesseract.image_to_string(
                    image,
                    lang='eng',
                    config='--psm 6'
                )
                return english_text, 'en', None, None
                
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return source_text, None, None, None
    else:
        # Regular invoice processing
        text = pytesseract.image_to_string(image, lang='eng')
        return text, None, None, None

def setup_ocr_pipeline(
    input_path: Path,
    output_dir: Path,
    visualization_dir: Optional[Path] = None,
    model_name: str = 'facebook/detr-resnet-50'
) -> OcrFn:
    """Factory function that sets up OCR pipeline with real tools
    
    Args:
        input_path: Path to input file or directory
        output_dir: Directory for OCR text output
        visualization_dir: Optional directory for saving visualizations
        model_name: DETR model to use
    """
    # Only create directories for output and visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    if visualization_dir:
        visualization_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize DETR models
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    translator = GoogleTranslator()
    
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Resize for better text detection
        return cv2.resize(denoised, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    def save_visualization(image: np.ndarray, confidence_data: dict, doc_id: str) -> None:
        """Save OCR visualization to file"""
        if not visualization_dir:
            return
            
        plt.figure(figsize=(20, 10))
        
        # Original image with OCR boxes
        plt.subplot(1, 2, 1)
        vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        
        n_boxes = len(confidence_data['text'])
        for i in range(n_boxes):
            if int(confidence_data['conf'][i]) > 60:
                (x, y, w, h) = (
                    confidence_data['left'][i],
                    confidence_data['top'][i],
                    confidence_data['width'][i],
                    confidence_data['height'][i]
                )
                
                color = (0, 255, 0)
                if any(c.isdigit() for c in confidence_data['text'][i]):
                    color = (0, 0, 255)
                    
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis_image, confidence_data['text'][i],
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        plt.imshow(vis_image)
        plt.title("OCR Detection Visualization")
        plt.axis('off')
        
        # Confidence heatmap
        plt.subplot(1, 2, 2)
        confidence_map = np.zeros((image.shape[0], image.shape[1]))
        for i in range(n_boxes):
            if int(confidence_data['conf'][i]) > 0:
                (x, y, w, h) = (
                    confidence_data['left'][i],
                    confidence_data['top'][i],
                    confidence_data['width'][i],
                    confidence_data['height'][i]
                )
                confidence_map[y:y+h, x:x+w] = confidence_data['conf'][i]
        
        plt.imshow(confidence_map, cmap='hot')
        plt.title("OCR Confidence Heatmap")
        plt.colorbar(label='Confidence')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(visualization_dir / f"{doc_id}_visualization.png")
        plt.close()
    
    def process_document(input_path: Path) -> OCRResult:
        """Process a single document with real OCR"""
        try:
            start_time = time.time()
            
            # Validate input
            if not input_path.exists():
                raise FileNotFoundError(f"File not found: {input_path}")
            
            if input_path.is_dir():
                raise IsADirectoryError(f"Expected file, got directory: {input_path}")
            
            # Generate document ID
            doc_id = f"DOC_{uuid.uuid4().hex[:8]}"
            
            # Handle PDF input for POST documents
            if input_path.suffix.lower() == '.pdf':
                logger.debug(f"Converting PDF to image: {input_path}")
                try:
                    # Convert PDF to image
                    pages = convert_from_path(input_path)
                    if not pages:
                        raise ValueError("Could not convert PDF to image")
                    logger.debug(f"Successfully converted PDF, got {len(pages)} pages")
                    image = np.array(pages[0])  # Process first page
                except Exception as e:
                    logger.error(f"PDF conversion failed: {e}")
                    raise
            else:
                # Regular image processing
                logger.debug(f"Processing image file: {input_path}")
                image = cv2.imread(str(input_path))
            
            if image is None:
                raise ValueError(f"Could not read document at {input_path}")
            
            processed_image = preprocess_image(image)
            
            # Process based on document type
            doc_type = detect_document_type(input_path.name)
            
            if doc_type == DocumentType.POST:
                # Use German OCR and translation
                text, source_lang, translated_text, translation_conf = process_text_with_translation(
                    processed_image, doc_type, translator
                )
            else:
                # Regular invoice processing
                text = pytesseract.image_to_string(processed_image)
                source_lang = None
                translated_text = None
                translation_conf = None
            
            # Calculate confidence score
            confidence_data = pytesseract.image_to_data(
                processed_image, 
                output_type=pytesseract.Output.DICT
            )
            
            valid_confidences = [c for c in confidence_data['conf'] if c > 0]
            confidence_score = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0
            
            # Save visualization
            save_visualization(processed_image, confidence_data, doc_id)
            
            # Save OCR text
            output_path = output_dir / f"{doc_id}.txt"
            output_text = text
            if translated_text:
                output_text = f"Original Text:\n{text}\n\nTranslated Text:\n{translated_text}"
            output_path.write_text(output_text)
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                document_id=doc_id,
                input_path=input_path,
                output_path=output_path,
                document_type=doc_type,
                processing_status=ProcessingStatus.COMPLETED,
                raw_text=text,
                confidence_score=confidence_score / 100,
                processing_time=processing_time,
                source_language=source_lang,
                translated_text=translated_text,
                translation_confidence=translation_conf
            )
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return OCRResult(
                document_id=f"DOC_{uuid.uuid4().hex[:8]}",
                input_path=input_path,
                output_path=output_dir / "error.txt",
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e),
                document_type=DocumentType.OTHER
            )
    
    return process_document 

def preprocess_financial_document(text: str) -> str:
    """Clean and format financial document text"""
    
    # 1. Protect financial identifiers
    protected_patterns = [
        (r'(IBAN\s*[A-Z0-9\s]+)', r' \1 '),  # IBANs
        (r'(BIC\s*[A-Z0-9]+)', r' \1 '),     # BICs
        (r'(VAT\s*[A-Z0-9]+)', r' \1 '),     # VAT numbers
        (r'(\d{2}/\d{3}/\d{5})', r' \1 '),   # Tax reference numbers
    ]
    
    # 2. Clean formatting artifacts
    cleanup_patterns = [
        (r'[_—]{2,}', ' '),        # Repeated separators
        (r'\s+', ' '),             # Multiple spaces
    ]
    
    # 3. Structure preservation
    structure_patterns = [
        (r'(\d+[.,]\d{2})', r' \1 '),  # Currency amounts
        (r'([A-Z]{2}\d{2}.*)', r' \1 ') # Reference numbers starting with country code
    ]
    
    # Apply all patterns
    for pattern, replacement in (protected_patterns + cleanup_patterns + structure_patterns):
        text = re.sub(pattern, replacement, text)
    
    return text.strip() 