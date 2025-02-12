from pathlib import Path
import time
import uuid
from typing import Optional
import logging
from dotenv import load_dotenv

from .data_model import OCRResult, ProcessingStatus, DocumentType, OcrFn
from .tools.OCR_tools import (
    load_document,
    preprocess_image,
    perform_ocr,
    detect_layout,
    save_visualization
)
from .tools.multilingual import (
    detect_and_translate,
    preprocess_financial_text,
    save_multilingual_output
)

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

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

def setup_ocr_pipeline(
    input_path: Path,
    output_dir: Path,
    visualization_dir: Optional[Path] = None
) -> OcrFn:
    """Factory function that sets up OCR pipeline"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    if visualization_dir:
        visualization_dir.mkdir(parents=True, exist_ok=True)
    
    def process_document(input_path: Path) -> OCRResult:
        """Process a single document with OCR"""
        try:
            start_time = time.time()
            doc_id = f"DOC_{uuid.uuid4().hex[:8]}"
            doc_type = detect_document_type(input_path.name)
            
            # Load and preprocess image
            image = load_document(input_path)
            processed_image = preprocess_image(image)
            
            # Perform OCR based on document type
            if doc_type == DocumentType.POST:
                # German OCR with translation
                text, confidence = perform_ocr(processed_image, lang='deu')
                text = preprocess_financial_text(text)
                original_text, translated_text, translation_conf = detect_and_translate(text)
            else:
                # Regular English OCR
                text, confidence = perform_ocr(processed_image)
                original_text, translated_text, translation_conf = text, None, None
            
            # Detect layout for visualization
            layout = detect_layout(image)
            
            # Save outputs
            output_path = output_dir / f"{doc_id}.txt"
            save_multilingual_output(original_text, translated_text, output_path)
            
            if visualization_dir:
                vis_path = visualization_dir / f"{doc_id}_visualization.png"
                save_visualization(image, text, confidence, layout, vis_path)
            
            # Create result object
            return OCRResult(
                document_id=doc_id,
                input_path=input_path,
                output_path=output_path,
                document_type=doc_type,
                processing_status=ProcessingStatus.COMPLETED,
                raw_text=original_text,
                confidence_score=confidence,
                processing_time=time.time() - start_time,
                source_language='de' if translated_text else None,
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