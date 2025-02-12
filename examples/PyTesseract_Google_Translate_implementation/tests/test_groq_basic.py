import pytest
import asyncio
from pathlib import Path
import logging

from src.llms.groq import create_groq_client, process_with_groq
from src.data_model import (
    OCRResult,
    ProcessingStatus,
    DocumentType,
    LLMStatus
)

@pytest.mark.asyncio
async def test_groq_basic_connectivity():
    """Test basic Groq API connectivity and processing"""
    # Setup logging
    log_file = Path("debug_groq.log")
    if log_file.exists():
        log_file.unlink()
    
    logger = logging.getLogger("groq_test")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
    # Test directories
    fixtures_dir = Path("./tests/fixtures")
    ocr_txt_dir = fixtures_dir / "ocr_txt_files"
    json_output_dir = fixtures_dir / "json_output"
    
    # Test both POST and invoice documents
    for doc_type in [DocumentType.INVOICE, DocumentType.POST]:
        # Get appropriate test file
        test_files = list((ocr_txt_dir / doc_type.value).glob("*.txt"))
        assert len(test_files) > 0, f"No {doc_type.value} test files found"
        
        test_file = test_files[0]
        logger.debug(f"Testing {doc_type.value} with file: {test_file}")
        
        content = test_file.read_text()
        logger.debug(f"File content (first 100 chars): {content[:100]}")
        
        # Create Groq client
        client, config = await create_groq_client()
        logger.debug("Created Groq client")
        
        # Create test OCR result
        ocr_result = OCRResult(
            document_id=f"TEST_{doc_type.value.upper()}",
            input_path=fixtures_dir / f"data/{doc_type.value}.pdf",
            output_path=test_file,
            document_type=doc_type,
            processing_status=ProcessingStatus.COMPLETED,
            raw_text=content
        )
        
        # Process document
        result = await process_with_groq(
            client=client,
            config=config,
            ocr_input=ocr_result,
            json_output_dir=json_output_dir
        )
        
        logger.debug(f"Process result: {result}")
        
        # Verify output
        assert result.processing_status == LLMStatus.COMPLETED
        assert result.json_content is not None
        assert result.document_type == doc_type
        
        # Verify expected fields based on document type
        if doc_type == DocumentType.POST:
            assert "document_identification" in result.json_content
            assert "tax_office_information" in result.json_content
        else:
            assert "invoice_details" in result.json_content
            assert "amounts" in result.json_content
        
    # Verify log file
    assert log_file.exists()
    assert log_file.read_text()
    
    return result 