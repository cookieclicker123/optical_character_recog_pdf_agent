from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import cv2

class DonutInvoiceProcessor:
    def __init__(self, model_dir="models/donut_invoice"):
        # Check if the model is already cached locally
        if not os.path.exists(model_dir):
            print("Downloading model for the first time...")
            self.save_model(model_dir)
        else:
            print("Loading model from local cache...")

        # Load the cached model and processor
        self.processor = DonutProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    def save_model(self, model_dir):
        """Download and save the model and processor to the specified directory."""
        # Use the invoice-specific model instead of base
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

        # Save to local directory
        processor.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        print(f"Model saved to {model_dir}")

    def process_invoice(self, image_path):
        """Processes the invoice image with detailed debugging output"""
        # Load and show original image dimensions
        image = Image.open(image_path).convert("RGB")
        print(f"\nOriginal Image Size: {image.size}")
        
        # Convert to numpy for visualization
        np_image = np.array(image)
        
        # Show preprocessing steps
        print("\n=== Preprocessing Steps ===")
        print("1. Image loaded and converted to RGB")
        
        # Get processor's expected input size
        processor_config = self.processor.feature_extractor.size
        print(f"2. Processor expected size: {processor_config}")
        
        # Add task prompt and process
        task_prompt = "<s_cord-v2>"
        print(f"3. Using task prompt: {task_prompt}")
        
        # Get processor output
        inputs = self.processor(images=image, text=task_prompt, return_tensors="pt")
        print("4. Image processed by Donut processor")
        
        print("\n=== Model Generation ===")
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,  # Increased from 1 to 4
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True
        )
        
        # Get raw text output
        predicted_text = self.processor.batch_decode(outputs.sequences)[0]
        print("\n=== Raw Model Output ===")
        print(predicted_text)
        
        # Show token-by-token output
        print("\n=== Token-by-Token Output ===")
        tokens = self.processor.tokenizer.convert_ids_to_tokens(outputs.sequences[0])
        for i, token in enumerate(tokens):
            if token not in ['<s>', '</s>', '<pad>']:
                print(f"Token {i}: {token}")
        
        try:
            # Clean and parse JSON
            print("\n=== JSON Parsing Attempt ===")
            cleaned_text = predicted_text.replace("<s>", "").replace("</s>", "").strip()
            print(f"Cleaned text: {cleaned_text}")
            
            if cleaned_text:
                structured_data = json.loads(cleaned_text)
            else:
                raise ValueError("Empty output from model")
            
            return structured_data, np_image
            
        except json.JSONDecodeError as e:
            print(f"\nJSON parsing error: {str(e)}")
            print(f"Attempted to parse text: '{cleaned_text}'")
            return {}, np_image

    def visualize_results(self, image, structured_data):
        """Visualizes both the image processing steps and results"""
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(121)
        plt.imshow(image)
        plt.title("Original Invoice")
        plt.axis('off')
        
        # Processed image with detected regions
        plt.subplot(122)
        # Convert back to RGB if needed
        if len(image.shape) == 2:
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_img = image.copy()
            
        # Add visualization of what Donut is looking at
        # This will show the regions Donut is processing
        plt.imshow(display_img)
        plt.title("Donut Processing Regions")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print structured data
        print("\n=== Extracted Data ===")
        if structured_data:
            print(json.dumps(structured_data, indent=4))
        else:
            print("No structured data extracted")

# Main function
if __name__ == "__main__":
    # Path to the invoice image you uploaded
    image_path = "./OCR_examples/data/invoice.png"

    # Initialize the processor with caching
    processor = DonutInvoiceProcessor()

    try:
        # Process the invoice
        structured_data, invoice_image = processor.process_invoice(image_path)

        # Visualize the results
        processor.visualize_results(invoice_image, structured_data)

        # Optional: Save the structured data to a JSON file
        with open("output_invoice_data.json", "w") as json_file:
            json.dump(structured_data, json_file, indent=4)

    except Exception as e:
        print(f"Error processing invoice: {str(e)}")
