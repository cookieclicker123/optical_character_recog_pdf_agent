from pathlib import Path
from typing import Dict, Any, Tuple
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import logging
from transformers import DetrImageProcessor, DetrForObjectDetection
from pdf2image import convert_from_path


logger = logging.getLogger(__name__)

def load_document(input_path: Path) -> np.ndarray:
    """Load document from file, handling both images and PDFs"""
    if input_path.suffix.lower() == '.pdf':
        logger.debug(f"Converting PDF to image: {input_path}")
        try:
            pages = convert_from_path(input_path)
            if not pages:
                raise ValueError("Could not convert PDF to image")
            return np.array(pages[0])  # Process first page
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    else:
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not read image at {input_path}")
        return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR results"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced)
    return cv2.resize(denoised, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

def perform_ocr(image: np.ndarray, lang: str = 'eng') -> Tuple[str, float]:
    """Perform OCR on image and return text with confidence score"""
    text = pytesseract.image_to_string(image, lang=lang)
    
    # Get confidence data
    confidence_data = pytesseract.image_to_data(
        image, 
        output_type=pytesseract.Output.DICT
    )
    
    # Calculate confidence score
    valid_confidences = [c for c in confidence_data['conf'] if c > 0]
    confidence_score = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0
    
    return text, confidence_score / 100

def detect_layout(image: np.ndarray, model_name: str = 'facebook/detr-resnet-50') -> Dict[str, Any]:
    """Detect document layout using DETR"""
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Convert outputs to useful format
    target_sizes = [(image.shape[0], image.shape[1])]
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes,
        threshold=0.7
    )[0]
    
    return {
        'boxes': results['boxes'].tolist(),
        'labels': results['labels'].tolist(),
        'scores': results['scores'].tolist()
    }

def save_visualization(
    image: np.ndarray,
    text: str,
    confidence: float,
    layout: Dict[str, Any],
    output_path: Path
) -> None:
    """Save visualization of OCR and layout detection results"""
    plt.figure(figsize=(20, 10))
    
    # Original image with detections
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw layout boxes
    for box, score in zip(layout['boxes'], layout['scores']):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, color='red', linewidth=2
        ))
    
    plt.title(f"OCR Confidence: {confidence:.2%}")
    plt.axis('off')
    
    # Save visualization
    plt.savefig(output_path)
    plt.close()
