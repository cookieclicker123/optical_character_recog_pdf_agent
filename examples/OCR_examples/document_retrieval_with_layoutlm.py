from transformers import LayoutLMv3Processor
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class LayoutLMInvoiceProcessor:
    def __init__(self):
        # Initialize LayoutLMv3 processor only
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

    def process_invoice(self, image_path):
        """Process invoice and return OCR text with layout visualization"""
        image = Image.open(image_path).convert("RGB")
        
        print("\n=== Processing Steps ===")
        print("1. Image loaded and converted to RGB")
        
        encoded_inputs = self.processor(
            image,
            return_tensors="pt"
        )
        
        print("2. OCR and layout analysis completed")
        
        # Get words and their positions - use decode instead of convert_ids_to_tokens
        input_ids = encoded_inputs.input_ids[0]
        boxes = encoded_inputs.bbox[0].tolist()
        
        # Decode each token properly
        words = []
        current_tokens = []
        
        for id in input_ids:
            token = self.processor.tokenizer.decode([id])
            if token not in ['<s>', '</s>', '<pad>']:
                words.append(token.strip())
        
        # Sort words by vertical position first (y-coordinate), then horizontal (x-coordinate)
        word_boxes = list(zip(words, boxes))
        
        # Group by lines (words with similar y-coordinates)
        line_height_threshold = 15
        current_line = []
        lines = []
        
        for i, (word, box) in enumerate(word_boxes):
            if not current_line:
                current_line.append((word, box))
            else:
                prev_y = current_line[-1][1][1]  # y-coordinate of previous word
                curr_y = box[1]  # y-coordinate of current word
                
                if abs(curr_y - prev_y) <= line_height_threshold:
                    current_line.append((word, box))
                else:
                    # Sort words in current line by x-coordinate
                    current_line.sort(key=lambda x: x[1][0])
                    lines.append(current_line)
                    current_line = [(word, box)]
        
        if current_line:
            current_line.sort(key=lambda x: x[1][0])
            lines.append(current_line)
        
        # Print formatted text
        print("\n=== Formatted OCR Text ===")
        formatted_text = ""
        for line in lines:
            line_text = " ".join(word for word, _ in line if word.strip())
            if line_text.strip():  # Only add non-empty lines
                formatted_text += line_text + "\n"
        print(formatted_text)
        
        return formatted_text, self.visualize_layout(image, encoded_inputs)

    def visualize_layout(self, image, encoded_inputs):
        """Visualize the detected layout and text regions"""
        np_image = np.array(image)
        
        # Draw boxes for each detected text region
        for box in encoded_inputs.bbox[0]:
            x1, y1, x2, y2 = box.tolist()
            cv2.rectangle(
                np_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
        
        return np_image

    def visualize_results(self, image):
        """Visualize detected regions"""
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Text Regions")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    processor = LayoutLMInvoiceProcessor()
    
    try:
        # Process invoice
        ocr_text, processed_image = processor.process_invoice("./OCR_examples/data/invoice3.png")
        
        # Visualize results
        processor.visualize_results(processed_image)
        
    except Exception as e:
        print(f"Error processing invoice: {str(e)}") 