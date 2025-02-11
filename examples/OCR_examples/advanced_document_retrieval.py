from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection

@dataclass
class ProcessedImage:
    """Data class to hold processed image and its OCR text"""
    image: np.ndarray
    text: str
    confidence_data: dict

def setup_models(model_name: str = 'facebook/detr-resnet-50') -> Tuple[DetrImageProcessor, DetrForObjectDetection]:
    """Initialize the DETR models"""
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    return processor, model

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

def process_document(image_path: Path) -> Optional[ProcessedImage]:
    """Process document and return OCR results"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Preprocess
    processed = preprocess_image(image)
    
    # Get OCR data
    ocr_data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    
    # Extract text
    text = pytesseract.image_to_string(processed)
    
    return ProcessedImage(
        image=processed,
        text=text,
        confidence_data=ocr_data
    )

def visualize_detection(processed: ProcessedImage) -> None:
    """Visualize OCR detection and confidence"""
    plt.figure(figsize=(20, 10))
    
    # Original image with OCR boxes
    plt.subplot(1, 2, 1)
    vis_image = cv2.cvtColor(processed.image.copy(), cv2.COLOR_GRAY2RGB)
    
    # Draw boxes around detected text
    n_boxes = len(processed.confidence_data['text'])
    for i in range(n_boxes):
        if int(processed.confidence_data['conf'][i]) > 60:  # Confidence threshold
            (x, y, w, h) = (
                processed.confidence_data['left'][i], 
                processed.confidence_data['top'][i],
                processed.confidence_data['width'][i], 
                processed.confidence_data['height'][i]
            )
            
            # Color based on content type
            color = (0, 255, 0)  # Default green
            detected_text = processed.confidence_data['text'][i]
            
            # Numbers in red
            if any(c.isdigit() for c in detected_text):
                color = (0, 0, 255)
                
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis_image, detected_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.imshow(vis_image)
    plt.title("OCR Detection Visualization")
    plt.axis('off')
    
    # Confidence heatmap
    plt.subplot(1, 2, 2)
    confidence_map = np.zeros((processed.image.shape[0], processed.image.shape[1]))
    for i in range(n_boxes):
        if int(processed.confidence_data['conf'][i]) > 0:
            (x, y, w, h) = (
                processed.confidence_data['left'][i], 
                processed.confidence_data['top'][i],
                processed.confidence_data['width'][i], 
                processed.confidence_data['height'][i]
            )
            confidence_map[y:y+h, x:x+w] = processed.confidence_data['conf'][i]
    
    plt.imshow(confidence_map, cmap='hot')
    plt.title("OCR Confidence Heatmap")
    plt.colorbar(label='Confidence')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_ocr_text(text: str, output_path: Path) -> None:
    """Save OCR text to file"""
    output_path.write_text(text)

def main():
    # Setup paths
    input_path = Path("./OCR_examples/data/invoice3.png")
    output_path = Path("./OCR_examples/output/ocr_text.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process document
        processed = process_document(input_path)
        
        # Save OCR text
        save_ocr_text(processed.text, output_path)
        print(f"OCR text saved to {output_path}")
        
        # Visualize results
        visualize_detection(processed)
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main() 