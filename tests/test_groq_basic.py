import pytest
import asyncio
from pathlib import Path
import logging

from src.llms.groq import create_groq_client, process_with_groq
from src.data_model import (
    OCRResult,
    ProcessingStatus,
    DocumentType
)

@pytest.mark.asyncio
async def test_groq_basic_connectivity():
    """Test basic Groq API connectivity and logging"""
    # Setup logging
    log_file = Path("debug_groq.log")
    if log_file.exists():
        log_file.unlink()
    
    logger = logging.getLogger("groq_test")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
    # Test file paths
    fixtures_dir = Path("./tests/fixtures")
    ocr_txt_dir = fixtures_dir / "ocr_txt_files"
    json_output_dir = fixtures_dir / "json_output"
    
    # Verify OCR text file exists
    ocr_files = list(ocr_txt_dir.glob("*.txt"))
    assert len(ocr_files) > 0, "No OCR text files found in fixtures"
    
    test_file = ocr_files[0]
    logger.debug(f"Testing with OCR file: {test_file}")
    
    # Read content
    content = test_file.read_text()
    logger.debug(f"File content (first 100 chars): {content[:100]}")
    assert content, "OCR file is empty"
    
    # Test API connection
    client, config = await create_groq_client()
    logger.debug("Created Groq client")
    
    # Create simple test OCR result
    ocr_result = OCRResult(
        document_id="TEST_DOC",
        input_path=fixtures_dir / "data/invoice.png",
        output_path=test_file,
        document_type=DocumentType.INVOICE,
        processing_status=ProcessingStatus.COMPLETED,
        raw_text=content
    )
    
    # Simple system prompt for testing
    test_prompt = "Extract invoice information and return as JSON."
    
    # Process single document
    result = await process_with_groq(
        client=client,
        config=config,
        ocr_input=ocr_result,
        json_output_dir=json_output_dir,
        system_prompt=test_prompt
    )
    
    logger.debug(f"Process result: {result}")
    
    # Verify log file was created and has content
    assert log_file.exists(), "Log file wasn't created"
    log_content = log_file.read_text()
    assert log_content, "Log file is empty"
    
    return result  # For manual inspection if needed 