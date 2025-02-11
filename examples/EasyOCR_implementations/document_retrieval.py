import easyocr
import cv2
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import ssl
import requests
from urllib.error import ContentTooShortError
import time
import numpy as np

def setup_environment():
    """Setup SSL context and create necessary directories."""
    # Fix SSL certificate verification
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Create directories if they don't exist
    os.makedirs("./EasyOCR_implementations/data", exist_ok=True)
    os.makedirs("./EasyOCR_implementations/output", exist_ok=True)
    os.makedirs("./EasyOCR_implementations/models", exist_ok=True)

def download_with_retry(url, save_path, max_retries=3):
    """Download file with retry mechanism."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")
            print(f"Download attempt {attempt + 1} failed, retrying in 5 seconds...")
            time.sleep(5)
    return False

def initialize_reader():
    """Initialize EasyOCR reader with model caching and retry mechanism."""
    # Set model storage directory
    model_storage_directory = os.path.join(os.getcwd(), 'models')
    
    max_retries = 6  # Increased from 3 to 6 retries
    for attempt in range(max_retries):
        try:
            # Set a longer timeout for the download
            reader = easyocr.Reader(
                ['en'], 
                model_storage_directory=model_storage_directory,
                download_enabled=True,
                verbose=True
            )
            return reader
        except ContentTooShortError:
            if attempt == max_retries - 1:
                raise Exception("Failed to download model after multiple attempts")
            print(f"Download attempt {attempt + 1} failed, retrying...")
            # Clean up any partial downloads
            detection_model_path = os.path.join(model_storage_directory, 'craft_mlt_25k.pth')
            if os.path.exists(detection_model_path):
                os.remove(detection_model_path)
            time.sleep(5)  # Wait 5 seconds before retrying

def preprocess_image(image_path, scale_factor=2.0):
    """Enhanced preprocessing with multi-scale processing for small text."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create upscaled version for small text
    height, width = gray.shape
    upscaled = cv2.resize(gray, (int(width * scale_factor), int(height * scale_factor)), 
                         interpolation=cv2.INTER_CUBIC)
    
    # Process both original and upscaled images
    images = {
        'original': process_single_scale(gray),
        'upscaled': process_single_scale(upscaled)
    }
    
    return images

def process_single_scale(img):
    """Process a single scale image."""
    # Simple contrast enhancement
    enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
    
    # Light denoising that preserves text edges
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Mild sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def get_skew_angle(image):
    """Detect skew angle of the document."""
    # Detect edges
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Use Hough transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:  # Filter out vertical lines
                angles.append(angle)
        
        if angles:
            return np.median(angles)
    return 0

def rotate_image(image, angle):
    """Rotate the image by given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), 
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    return rotated

def perform_ocr(images, reader):
    """Perform OCR with multi-scale processing."""
    # Process original scale
    results_original = reader.readtext(
        images['original'],
        paragraph=False,
        batch_size=4,
        text_threshold=0.5,
        link_threshold=0.3,
        low_text=0.3,
        mag_ratio=1.5
    )
    
    # Process upscaled version for small text
    results_upscaled = reader.readtext(
        images['upscaled'],
        paragraph=False,
        batch_size=4,
        text_threshold=0.3,  # Lower threshold for small text
        link_threshold=0.3,
        low_text=0.3,
        mag_ratio=1.0  # No additional magnification needed
    )
    
    # Combine results with priority to higher confidence
    combined_results = merge_results(results_original, results_upscaled)
    
    return combined_results

def merge_results(original_results, upscaled_results):
    """Merge results from different scales, prioritizing higher confidence."""
    merged = []
    used_texts = set()
    
    # Helper function to normalize bounding box coordinates
    def normalize_bbox(bbox, is_upscaled=False):
        if is_upscaled:
            return [[x/2, y/2] for x, y in bbox]
        return bbox
    
    # Add original results first
    for bbox, text, conf in original_results:
        if len(text.strip()) > 0:  # Skip empty text
            merged.append((bbox, text, conf))
            used_texts.add(text.lower())
    
    # Add upscaled results if they're new or have higher confidence
    for bbox, text, conf in upscaled_results:
        text_lower = text.lower()
        if len(text.strip()) > 0 and (
            text_lower not in used_texts or 
            conf > 0.8  # High confidence threshold for upscaled results
        ):
            merged.append((normalize_bbox(bbox, True), text, conf))
            used_texts.add(text_lower)
    
    return merged

def save_text_results(results, output_txt_path):
    """Save extracted text to a .txt file."""
    with open(output_txt_path, 'w') as file:
        for _, text, confidence in results:
            file.write(f"{text} (Confidence: {confidence:.2f})\n")

def generate_pdf_report(results, output_pdf_path):
    """Generate a PDF report with extracted text."""
    pdf = canvas.Canvas(output_pdf_path, pagesize=letter)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 750, "OCR Report")
    pdf.drawString(100, 730, "-" * 50)

    y_position = 700
    for _, text, confidence in results:
        pdf.drawString(100, y_position, f"Text: {text}")
        pdf.drawString(100, y_position - 20, f"Confidence: {confidence:.2f}")
        y_position -= 40
        if y_position < 100:  # Start a new page if there's not enough space
            pdf.showPage()
            y_position = 750

    pdf.save()

def visualize_results(image, results):
    """Display the image with bounding boxes for detected text."""
    for (bbox, text, confidence) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main function to tie everything together
def main():
    # Setup environment first
    setup_environment()
    
    image_path = "./EasyOCR_implementations/data/invoice.png"
    output_txt_path = "./EasyOCR_implementations/output/extracted_text.txt"
    output_pdf_path = "./EasyOCR_implementations/output/ocr_report.pdf"

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    try:
        # Initialize EasyOCR reader with custom model storage
        reader = initialize_reader()
        
        # Preprocess the image
        preprocessed_images = preprocess_image(image_path)

        # Perform OCR
        results = perform_ocr(preprocessed_images, reader)

        # Sort results by vertical position for more logical output
        results.sort(key=lambda x: x[0][0][1])  # Sort by y-coordinate
        
        # Save and generate outputs
        save_text_results(results, output_txt_path)
        generate_pdf_report(results, output_pdf_path)

        # Visualize results
        visualize_results(cv2.imread(image_path), results)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
