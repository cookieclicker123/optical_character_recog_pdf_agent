import pytest
from pathlib import Path
import shutil
import json
from datetime import datetime, timedelta
from src.mock_llm_information_grouping import setup_llm_pipeline
from src.data_model import (
    OCRResult, LLMResult, ProcessingStatus, DocumentType, 
    LLMStatus
)

@pytest.fixture
def test_dirs():
    """Setup and teardown test directories"""
    base_dir = Path("tests/test_data")
    ocr_dir = base_dir / "ocr_output"
    json_dir = base_dir / "json_output"
    
    # Setup
    ocr_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    
    yield ocr_dir, json_dir
    
    # Teardown
    shutil.rmtree(base_dir)

@pytest.fixture
def llm_processor(test_dirs):
    """Create LLM processor function"""
    _, json_dir = test_dirs
    return setup_llm_pipeline(json_dir)

@pytest.fixture
def sample_ocr_result(test_dirs):
    """Create a sample OCR result"""
    ocr_dir, _ = test_dirs
    ocr_file = ocr_dir / "DOC_12345678.txt"
    ocr_file.write_text("Sample OCR text with invoice details")
    
    return OCRResult(
        document_id="DOC_12345678",
        input_path=Path("original.pdf"),
        output_path=ocr_file,
        document_type=DocumentType.INVOICE,
        processing_status=ProcessingStatus.COMPLETED,
        raw_text="Sample OCR text",
        confidence_score=0.95,
        processing_time=0.5
    )

def test_factory_function_setup(test_dirs):
    """Test factory function creates directory and returns LLMFn"""
    _, json_dir = test_dirs
    processor = setup_llm_pipeline(json_dir)
    
    assert json_dir.exists()
    assert callable(processor)

def test_successful_processing(llm_processor, sample_ocr_result):
    """Test successful document processing"""
    result = llm_processor(sample_ocr_result)
    
    assert isinstance(result, LLMResult)
    assert result.document_id == sample_ocr_result.document_id
    assert result.processing_status == LLMStatus.COMPLETED
    assert result.json_content is not None
    assert result.processing_time is not None
    assert result.tokens_used is not None
    assert result.error_message is None
    assert (datetime.now() - result.created_at) < timedelta(seconds=2)

def test_json_file_creation(llm_processor, sample_ocr_result, test_dirs):
    """Test JSON file is created with correct content"""
    _, json_dir = test_dirs
    result = llm_processor(sample_ocr_result)
    
    assert result.output_path.exists()
    assert result.output_path.parent == json_dir
    assert result.output_path.suffix == ".json"
    
    # Verify JSON content
    with open(result.output_path) as f:
        content = json.load(f)
    assert isinstance(content, dict)
    assert "invoice_details" in content
    assert content["invoice_details"]["invoice_number"].startswith("INV-")

def test_failed_ocr_handling(llm_processor, sample_ocr_result):
    """Test handling of failed OCR results"""
    failed_ocr = sample_ocr_result.copy(
        update={"processing_status": ProcessingStatus.FAILED}
    )
    
    result = llm_processor(failed_ocr)
    assert result.processing_status == LLMStatus.FAILED
    assert result.error_message is not None
    assert result.json_content is None

def test_missing_ocr_file(llm_processor, sample_ocr_result):
    """Test handling of missing OCR text file"""
    missing_file_ocr = sample_ocr_result.copy(
        update={"output_path": Path("nonexistent.txt")}
    )
    
    result = llm_processor(missing_file_ocr)
    assert result.processing_status == LLMStatus.FAILED
    assert "not found" in result.error_message.lower()

def test_result_immutability(llm_processor, sample_ocr_result):
    """Test LLMResult immutability"""
    result = llm_processor(sample_ocr_result)
    
    with pytest.raises(Exception):
        result.document_id = "new_id"

def test_processing_time_tracking(llm_processor, sample_ocr_result):
    """Test processing time is tracked correctly"""
    result = llm_processor(sample_ocr_result)
    
    assert isinstance(result.processing_time, float)
    assert result.processing_time > 0
    assert result.processing_time < 2

def test_json_structure(llm_processor, sample_ocr_result):
    """Test JSON output structure"""
    result = llm_processor(sample_ocr_result)
    
    required_fields = {
        "invoice_details", "amounts", "vendor", "line_items"
    }
    assert all(field in result.json_content for field in required_fields)
    
    # Test nested structure
    assert isinstance(result.json_content["line_items"], list)
    assert all(isinstance(item, dict) for item in result.json_content["line_items"])

def test_token_tracking(llm_processor, sample_ocr_result):
    """Test token usage tracking"""
    result = llm_processor(sample_ocr_result)
    
    assert isinstance(result.tokens_used, int)
    assert result.tokens_used > 0

def test_multiple_documents(llm_processor, sample_ocr_result, test_dirs):
    """Test processing multiple documents"""
    ocr_dir, _ = test_dirs
    results = []
    
    for i in range(3):
        ocr_file = ocr_dir / f"DOC_{i}.txt"
        ocr_file.write_text(f"Sample OCR text {i}")
        
        ocr_result = sample_ocr_result.copy(
            update={
                "document_id": f"DOC_{i}",
                "output_path": ocr_file
            }
        )
        results.append(llm_processor(ocr_result))
    
    assert all(r.processing_status == LLMStatus.COMPLETED for r in results)
    assert len({r.document_id for r in results}) == 3  # Unique IDs 