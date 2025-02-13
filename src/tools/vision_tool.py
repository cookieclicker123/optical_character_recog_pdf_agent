from pathlib import Path
import time
from typing import Dict, Any, List
import base64
from pdf2image import convert_from_path
from groq import Groq
from ..data_model import VisionTool, VisionToolFn
from utils.config import get_groq_config

def setup_vision_tool(config: Dict[str, Any] = None) -> VisionToolFn:
    """Factory function that returns a vision tool processing function
    
    Args:
        config: Optional custom configuration, defaults to groq_config
        
    Returns:
        Function that takes a Path and prompt and returns a VisionTool result
    """
    if config is None:
        config = get_groq_config()
    
    # Initialize Groq client
    client = Groq(api_key=config['api_key'])
    
    def pdf_to_images(pdf_path: Path, output_dir: Path = None) -> List[Path]:
        """Convert PDF to images, one per page
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Optional directory to save images, defaults to pdf_path.parent/pdf_path.stem
            
        Returns:
            List of paths to generated images
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        # Create output directory if not specified
        if output_dir is None:
            output_dir = pdf_path.parent / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        image_paths = []
        
        # Save each page as an image
        for i, image in enumerate(images):
            image_path = output_dir / f"page_{i+1}.jpg"
            image.save(image_path, "JPEG")
            image_paths.append(image_path)
            
        return image_paths
    
    def encode_image(image_path: Path) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def process_image(input_path: Path, prompt: str) -> VisionTool:
        """Process a single image or PDF page with the vision model"""
        try:
            start_time = time.time()
            
            # Validate input
            if not input_path.exists():
                raise FileNotFoundError(f"File not found: {input_path}")
            
            if not prompt:
                raise ValueError("Prompt cannot be empty")
            
            # Handle PDF input
            if input_path.suffix.lower() == '.pdf':
                # Convert first page only for now
                image_paths = pdf_to_images(input_path)
                if not image_paths:
                    raise ValueError("No pages found in PDF")
                image_path = image_paths[0]  # Process first page
            else:
                image_path = input_path
            
            # Encode image
            base64_image = encode_image(image_path)
            
            # Prepare messages for the vision model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Call the vision model
            response = client.chat.completions.create(
                model=config['model_name'],
                messages=messages,
                temperature=config['temperature'],
                max_tokens=config['max_tokens']
            )
            
            # Extract response and token usage
            model_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Format response as markdown (assuming model returns structured text)
            markdown_response = model_response
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return VisionTool(
                prompt=prompt,
                response=model_response,
                markdown=markdown_response,
                tokens_used=tokens_used,
                processing_time=processing_time
            )
            
        except Exception as e:
            return VisionTool(
                prompt=prompt,
                response="",
                markdown="",
                tokens_used=0,
                processing_time=0.0,
                error_message=str(e)
            )
    
    return process_image 