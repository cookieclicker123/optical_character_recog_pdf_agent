import json
import uuid
import time
from typing import Dict, Any, List, Union
import httpx
from pathlib import Path
import logging

from ..data_model import (
    LLMResult,
    LLMBatch,
    LLMStatus,
    OCRResult,
    OCRBatch
)
from utils.config import get_groq_config

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("groq_api.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

async def create_groq_client():
    """Creates a connection to Groq API"""
    config = get_groq_config()
    client = httpx.AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
    )
    return client, config

async def process_with_groq(
    client: httpx.AsyncClient,
    config: Dict[str, Any],
    ocr_input: Union[OCRResult, OCRBatch],
    json_output_dir: Path,
    system_prompt: str
) -> Union[LLMResult, LLMBatch]:
    """Process OCR results through Groq API and return structured data
    
    Args:
        client: Initialized Groq client
        config: Groq configuration
        ocr_input: Single OCR result or batch of results
        json_output_dir: Directory for JSON output files
        system_prompt: System prompt for Groq
        
    Returns:
        LLMResult for single document or LLMBatch for multiple documents
    """
    
    async def process_single_document(ocr_result: OCRResult) -> LLMResult:
        """Process a single document with Groq"""
        start_time = time.time()
        
        try:
            logger.debug(f"\n{'='*50}\nProcessing document: {ocr_result.document_id}")
            
            # Verify OCR result
            if not ocr_result.output_path.exists():
                raise FileNotFoundError(f"OCR text file not found: {ocr_result.output_path}")
            
            # Read and log OCR text
            ocr_text = ocr_result.output_path.read_text()
            logger.debug(f"OCR Text Content (first 500 chars):\n{ocr_text[:500]}...")
            
            # Make API call
            response = await client.post(
                "/chat/completions",
                json={
                    "model": config["model_name"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": ocr_text}
                    ],
                    "temperature": config["temperature"],
                    "max_tokens": config["max_tokens"]
                }
            )
            
            # Log raw response
            logger.debug(f"API Response Status Code: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            
            # Extract content parts
            content = result["choices"][0]["message"]["content"].strip()
            logger.debug(f"Raw content:\n{content}")
            
            # Split thinking and JSON parts
            think_match = content.split("</think>")
            if len(think_match) > 1:
                reasoning = think_match[0].replace("<think>", "").strip()
                json_part = think_match[1].strip()
            else:
                reasoning = None
                json_part = content
            
            # Extract JSON from code blocks if present
            if "```json" in json_part:
                json_str = json_part.split("```json")[1].split("```")[0].strip()
            else:
                json_str = json_part.strip()
            
            logger.debug(f"Extracted JSON string:\n{json_str}")
            
            try:
                json_content = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON string:\n{json_str}")
                logger.error(f"JSON parse error: {str(e)}")
                raise
            
            # Save to file
            json_path = json_output_dir / f"{ocr_result.document_id}.json"
            json_path.write_text(json.dumps(json_content, indent=2))
            logger.debug(f"Saved JSON to: {json_path}")
            
            # Calculate metrics
            processing_time = time.time() - start_time
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            
            return LLMResult(
                document_id=ocr_result.document_id,
                input_path=ocr_result.output_path,
                output_path=json_path,
                processing_status=LLMStatus.COMPLETED,
                json_content=json_content,
                llm_reasoning=reasoning,
                processing_time=processing_time,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"Error processing document {ocr_result.document_id}: {str(e)}", exc_info=True)
            return LLMResult(
                document_id=ocr_result.document_id,
                input_path=ocr_result.output_path,
                output_path=json_output_dir / f"{ocr_result.document_id}_error.json",
                processing_status=LLMStatus.FAILED,
                error_message=str(e),
                document_type=ocr_result.document_type
            )
    
    # Handle single document
    if isinstance(ocr_input, OCRResult):
        return await process_single_document(ocr_input)
    
    # Handle batch processing
    batch_id = f"BATCH_{uuid.uuid4().hex[:8]}"
    results: List[LLMResult] = []
    processed = 0
    
    for ocr_result in ocr_input.results:
        llm_result = await process_single_document(ocr_result)
        results.append(llm_result)
        processed += 1 if llm_result.processing_status == LLMStatus.COMPLETED else 0
    
    return LLMBatch(
        batch_id=batch_id,
        results=results,
        total_documents=len(ocr_input.results),
        processed_documents=processed,
        status=LLMStatus.COMPLETED if processed == len(ocr_input.results) else LLMStatus.FAILED
    ) 