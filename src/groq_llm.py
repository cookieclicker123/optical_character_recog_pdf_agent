from pathlib import Path
from typing import Union, Optional
import logging

from .data_model import (
    OCRResult,
    OCRBatch,
    LLMResult,
    LLMBatch,
    LLMStatus,
    LLMFn
)
from .llms.groq import create_groq_client, process_with_groq
from .OCR import setup_ocr_pipeline
from .prompts.invoice_extraction_prompt import INVOICE_SYSTEM_PROMPT

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("groq_pipeline.log")
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

async def setup_groq_pipeline(
    json_output_dir: Path,
    ocr_output_dir: Optional[Path] = None,
    visualization_dir: Optional[Path] = None
) -> LLMFn:
    """Factory function that sets up the complete OCR + Groq pipeline
    
    Args:
        json_output_dir: Directory for JSON output files
        ocr_output_dir: Optional directory for OCR text output (if doing OCR)
        visualization_dir: Optional directory for visualizations
        
    Returns:
        Function that processes documents through OCR and Groq
    """
    logger.debug("Setting up Groq pipeline...")
    logger.debug(f"System Prompt loaded: {INVOICE_SYSTEM_PROMPT[:100]}...")  # Log first 100 chars
    
    # Create output directories
    json_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Groq client
    client, config = await create_groq_client()
    logger.debug(f"Groq client initialized with model: {config['model_name']}")
    
    async def process_documents(
        input_data: Union[Path, OCRResult, OCRBatch]
    ) -> Union[LLMResult, LLMBatch]:
        """Process documents through OCR (if needed) and Groq
        
        Args:
            input_data: Either:
                - Path to image file/directory (needs OCR)
                - OCRResult (already processed)
                - OCRBatch (multiple processed documents)
        """
        try:
            logger.debug(f"Processing input type: {type(input_data)}")
            
            # Case 1: Raw image input - needs OCR first
            if isinstance(input_data, Path):
                if ocr_output_dir is None:
                    raise ValueError("OCR output directory required for image processing")
                
                logger.debug(f"Processing raw image: {input_data}")
                # Setup and run OCR
                ocr_fn = setup_ocr_pipeline(
                    input_path=input_data,
                    output_dir=ocr_output_dir,
                    visualization_dir=visualization_dir
                )
                ocr_result = ocr_fn(input_data)
                
                # Process through Groq
                return await process_with_groq(
                    client=client,
                    config=config,
                    ocr_input=ocr_result,
                    json_output_dir=json_output_dir,
                    system_prompt=INVOICE_SYSTEM_PROMPT
                )
            
            # Case 2: Pre-processed OCR data
            logger.debug(f"Processing pre-processed OCR data")
            return await process_with_groq(
                client=client,
                config=config,
                ocr_input=input_data,
                json_output_dir=json_output_dir,
                system_prompt=INVOICE_SYSTEM_PROMPT
            )
            
        except Exception as e:
            logger.error(f"Error in process_documents: {str(e)}", exc_info=True)
            raise
    
    return process_documents