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


def test_verify_ocr_files():
    """Verify OCR files exist before running other tests"""
    fixtures_dir = Path("./tests/fixtures")
    
    # Check invoice files
    invoice_dir = fixtures_dir / "ocr_txt_files"
    print("\nAll files in invoice directory:")
    for f in invoice_dir.iterdir():
        print(f"  {f.name}")
    invoice_files = list(invoice_dir.glob("DOC_*.txt"))
    print(f"Files matching 'DOC_*.txt': {[f.name for f in invoice_files]}")
    assert len(invoice_files) > 0, f"No invoice files found in {invoice_dir}"
    
    # Check post files
    post_dir = fixtures_dir / "ocr_post_txt_files"
    print("\nAll files in post directory:")
    for f in post_dir.iterdir():
        print(f"  {f.name}")
    post_files = list(post_dir.glob("DOC_*.txt"))
    print(f"Files matching 'DOC_*.txt': {[f.name for f in post_files]}")
    assert len(post_files) > 0, f"No post files found in {post_dir}"

@pytest.fixture
def test_dirs():
    """Setup test directories"""
    fixtures_dir = Path("./tests/fixtures")
    return {
        'fixtures': fixtures_dir,
        'json_invoice_output': fixtures_dir / "json_invoice_output",
        'json_post_output': fixtures_dir / "json_post_output",
        'ocr_txt': {
            'invoice': fixtures_dir / "ocr_txt_files",
            'post': fixtures_dir / "ocr_post_txt_files"
        }
    }

@pytest.mark.asyncio
async def test_single_document_processing(test_dirs):
    """Test processing single documents through Groq using real OCR output"""
    for doc_type in [DocumentType.INVOICE, DocumentType.POST]:
        # Get correct paths for document type
        ocr_dir = test_dirs['ocr_txt'][doc_type.value]
        json_dir = test_dirs[f'json_{doc_type.value}_output']
        
        # Use existing OCR text file from fixtures - using DOC_ pattern
        ocr_txt_file = next(ocr_dir.glob("DOC_*.txt"))
        doc_id = ocr_txt_file.stem
        
        ocr_result = OCRResult(
            document_id=doc_id,
            input_path=test_dirs['fixtures'] / f"data/{doc_type.value}.png",
            output_path=ocr_txt_file,
            document_type=doc_type,
            processing_status=ProcessingStatus.COMPLETED,
            raw_text=ocr_txt_file.read_text()
        )
        
        # Setup and run pipeline with correct output directory
        pipeline = await setup_groq_pipeline(json_output_dir=json_dir)
        result = await pipeline(ocr_result)
        
        # Verify structure and files
        assert isinstance(result, LLMResult)
        assert result.document_id == doc_id
        assert result.processing_status == LLMStatus.COMPLETED
        assert result.json_content is not None
        assert result.tokens_used is not None
        assert result.processing_time is not None
        
        # Verify file creation
        json_path = json_dir / f"{doc_id}.json"
        assert json_path.exists()
        assert json_path == result.output_path

@pytest.mark.asyncio
async def test_batch_processing(test_dirs):
    """Test processing multiple OCR results as a batch using real OCR outputs"""
    # Get OCR text files for both document types using DOC_ pattern
    invoice_files = list(test_dirs['ocr_txt']['invoice'].glob("DOC_*.txt"))[:1]
    post_files = list(test_dirs['ocr_txt']['post'].glob("DOC_*.txt"))[:1]
    ocr_txt_files = invoice_files + post_files
    
    # Create batch from real files
    ocr_batch = OCRBatch(
        batch_id="BATCH_12345",
        results=[
            OCRResult(
                document_id=txt_file.stem,
                input_path=test_dirs['fixtures'] / f"data/{'post' if 'POST_' in txt_file.name else 'invoice'}.png",
                output_path=txt_file,
                document_type=DocumentType.POST if 'POST_' in txt_file.name else DocumentType.INVOICE,
                processing_status=ProcessingStatus.COMPLETED,
                raw_text=txt_file.read_text()
            )
            for txt_file in ocr_txt_files
        ],
        total_documents=len(ocr_txt_files),
        processed_documents=len(ocr_txt_files),
        status=ProcessingStatus.COMPLETED
    )
    
    # Process each document type separately
    for doc_type in [DocumentType.INVOICE, DocumentType.POST]:
        pipeline = await setup_groq_pipeline(
            json_output_dir=test_dirs[f'json_{doc_type.value}_output']
        )
        batch_results = [r for r in ocr_batch.results if r.document_type == doc_type]
        if batch_results:
            sub_batch = OCRBatch(
                batch_id=f"{ocr_batch.batch_id}_{doc_type.value}",
                results=batch_results,
                total_documents=len(batch_results),
                processed_documents=len(batch_results),
                status=ProcessingStatus.COMPLETED
            )
            result = await pipeline(sub_batch)
            
            # Verify batch structure
            assert isinstance(result, LLMBatch)
            assert result.total_documents == len(batch_results)
            assert result.processed_documents == len(batch_results)
            assert result.status == LLMStatus.COMPLETED
            assert len(result.results) == len(batch_results)
            
            # Verify individual results
            for llm_result, ocr_result in zip(result.results, batch_results):
                assert isinstance(llm_result, LLMResult)
                assert llm_result.document_id == ocr_result.document_id
                assert llm_result.processing_status == LLMStatus.COMPLETED
                assert llm_result.json_content is not None
                assert llm_result.tokens_used is not None
                assert llm_result.processing_time is not None
                
                # Verify file creation
                json_path = test_dirs[f'json_{doc_type.value}_output'] / f"{ocr_result.document_id}.json"
                assert json_path.exists()
                assert json_path == llm_result.output_path

@pytest.mark.asyncio
async def test_error_handling(test_dirs):
    """Test error handling in the pipeline for both document types"""
    for doc_type in [DocumentType.INVOICE, DocumentType.POST]:
        ocr_result = OCRResult(
            document_id=f"{doc_type.value.upper()}_ERROR",
            input_path=test_dirs['fixtures'] / "nonexistent.png",
            output_path=test_dirs['ocr_txt'][doc_type.value] / f"nonexistent_{doc_type.value}.txt",
            document_type=doc_type,
            processing_status=ProcessingStatus.COMPLETED
        )
        
        pipeline = await setup_groq_pipeline(json_output_dir=test_dirs[f'json_{doc_type.value}_output'])
        result = await pipeline(ocr_result)
        
        assert isinstance(result, LLMResult)
        assert result.processing_status == LLMStatus.FAILED
        assert result.error_message is not None
        assert result.document_type == doc_type