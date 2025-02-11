from pathlib import Path
import time
import uuid
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from typing import Optional
from transformers import DetrImageProcessor, DetrForObjectDetection

from .data_model import OCRResult, ProcessingStatus, DocumentType, OcrFn

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
            
            # Read and process image
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Could not read image at {input_path}")
            
            processed_image = preprocess_image(image)
            
            # Perform OCR
            confidence_data = pytesseract.image_to_data(
                processed_image, 
                output_type=pytesseract.Output.DICT
            )
            
            text = pytesseract.image_to_string(processed_image)
            
            # Calculate confidence score
            valid_confidences = [c for c in confidence_data['conf'] if c > 0]
            confidence_score = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0
            
            # Save visualization
            save_visualization(processed_image, confidence_data, doc_id)
            
            # Save OCR text with verification
            output_path = output_dir / f"{doc_id}.txt"
            output_path.write_text(text)
            
            # Verify write
            if not output_path.exists():
                raise IOError(f"Failed to create output file at {output_path}")
            
            # Double check content
            written_content = output_path.read_text()
            if not written_content:
                raise IOError(f"Output file is empty: {output_path}")
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                document_id=doc_id,
                input_path=input_path,
                output_path=output_path,
                document_type=DocumentType.INVOICE,  # TODO: Add real detection
                processing_status=ProcessingStatus.COMPLETED,
                raw_text=text,
                confidence_score=confidence_score / 100,  # Convert to 0-1 scale
                processing_time=processing_time
            )
            
        except Exception as e:
            return OCRResult(
                document_id=f"DOC_{uuid.uuid4().hex[:8]}",
                input_path=input_path,
                output_path=output_dir / "error.txt",
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e),
                document_type=DocumentType.OTHER
            )
    
    return process_document 