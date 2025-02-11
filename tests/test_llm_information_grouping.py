import pytest
import asyncio
from pathlib import Path

from src.groq_llm import setup_groq_pipeline
from src.data_model import (
    OCRResult,
    OCRBatch,
    LLMResult,
    LLMBatch,
    LLMStatus,
    ProcessingStatus,
    DocumentType
)

@pytest.fixture
def test_dirs():
    """Setup test directories"""
    fixtures_dir = Path("./tests/fixtures")
    return {
        'fixtures': fixtures_dir,
        'json_output': fixtures_dir / "json_output",
        'ocr_txt': fixtures_dir / "ocr_txt_files"
    }

@pytest.mark.asyncio
async def test_single_document_processing(test_dirs):
    """Test processing a single OCR result through Groq using real OCR output"""
    # Use existing OCR text file from fixtures
    ocr_txt_file = next(test_dirs['ocr_txt'].glob("DOC_*.txt"))
    doc_id = ocr_txt_file.stem
    
    ocr_result = OCRResult(
        document_id=doc_id,
        input_path=test_dirs['fixtures'] / "data/invoice.png",
        output_path=ocr_txt_file,
        document_type=DocumentType.INVOICE,
        processing_status=ProcessingStatus.COMPLETED,
        raw_text=ocr_txt_file.read_text()
    )
    
    # Setup and run pipeline
    pipeline = await setup_groq_pipeline(json_output_dir=test_dirs['json_output'])
    result = await pipeline(ocr_result)
    
    # Verify structure and files
    assert isinstance(result, LLMResult)
    assert result.document_id == doc_id
    assert result.processing_status == LLMStatus.COMPLETED
    assert result.json_content is not None
    assert result.tokens_used is not None
    assert result.processing_time is not None
    
    # Verify file creation
    json_path = test_dirs['json_output'] / f"{doc_id}.json"
    assert json_path.exists()
    assert json_path == result.output_path

@pytest.mark.asyncio
async def test_batch_processing(test_dirs):
    """Test processing multiple OCR results as a batch using real OCR outputs"""
    # Get all OCR text files from fixtures
    ocr_txt_files = list(test_dirs['ocr_txt'].glob("DOC_*.txt"))[:2]  # Use first 2 files
    
    # Create batch from real files
    ocr_batch = OCRBatch(
        batch_id="BATCH_12345",
        results=[
            OCRResult(
                document_id=txt_file.stem,
                input_path=test_dirs['fixtures'] / "data/invoice.png",
                output_path=txt_file,
                document_type=DocumentType.INVOICE,
                processing_status=ProcessingStatus.COMPLETED,
                raw_text=txt_file.read_text()
            )
            for txt_file in ocr_txt_files
        ],
        total_documents=len(ocr_txt_files),
        processed_documents=len(ocr_txt_files),
        status=ProcessingStatus.COMPLETED
    )
    
    # Setup and run pipeline
    pipeline = await setup_groq_pipeline(json_output_dir=test_dirs['json_output'])
    result = await pipeline(ocr_batch)
    
    # Verify batch structure
    assert isinstance(result, LLMBatch)
    assert result.total_documents == len(ocr_txt_files)
    assert result.processed_documents == len(ocr_txt_files)
    assert result.status == LLMStatus.COMPLETED
    assert len(result.results) == len(ocr_txt_files)
    
    # Verify individual results
    for llm_result, txt_file in zip(result.results, ocr_txt_files):
        assert isinstance(llm_result, LLMResult)
        assert llm_result.document_id == txt_file.stem
        assert llm_result.processing_status == LLMStatus.COMPLETED
        assert llm_result.json_content is not None
        assert llm_result.tokens_used is not None
        assert llm_result.processing_time is not None
        
        # Verify file creation
        json_path = test_dirs['json_output'] / f"{txt_file.stem}.json"
        assert json_path.exists()
        assert json_path == llm_result.output_path

@pytest.mark.asyncio
async def test_error_handling(test_dirs):
    """Test error handling in the pipeline"""
    ocr_result = OCRResult(
        document_id="DOC_ERROR",
        input_path=test_dirs['fixtures'] / "nonexistent.png",
        output_path=test_dirs['ocr_txt'] / "nonexistent.txt",
        document_type=DocumentType.INVOICE,
        processing_status=ProcessingStatus.COMPLETED
    )
    
    pipeline = await setup_groq_pipeline(json_output_dir=test_dirs['json_output'])
    result = await pipeline(ocr_result)
    
    assert isinstance(result, LLMResult)
    assert result.processing_status == LLMStatus.FAILED
    assert result.error_message is not None