import pytest
from pathlib import Path
import shutil
from datetime import datetime, timedelta
from src.mock_OCR import setup_ocr_pipeline
from src.data_model import ProcessingStatus, DocumentType, OCRResult

@pytest.fixture
def test_dirs():
    """Setup and teardown test directories"""
    base_dir = Path("tests/test_data")
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    
    # Setup
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yield input_dir, output_dir
    
    # Teardown
    shutil.rmtree(base_dir)

@pytest.fixture
def ocr_processor(test_dirs):
    """Create OCR processor function"""
    input_dir, output_dir = test_dirs
    return setup_ocr_pipeline(input_dir, output_dir)

def test_factory_function_setup(test_dirs):
    """Test factory function creates directories and returns OcrFn"""
    input_dir, output_dir = test_dirs
    processor = setup_ocr_pipeline(input_dir, output_dir)
    
    assert input_dir.exists()
    assert output_dir.exists()
    assert callable(processor)

def test_successful_processing(ocr_processor, test_dirs):
    """Test successful document processing"""
    input_dir, _ = test_dirs
    test_file = input_dir / "test_doc.pdf"
    test_file.touch()
    
    result = ocr_processor(test_file)
    
    assert isinstance(result, OCRResult)
    assert result.document_id.startswith("DOC_")
    assert result.processing_status == ProcessingStatus.COMPLETED
    assert result.input_path == test_file
    assert result.raw_text is not None
    assert result.confidence_score is not None
    assert result.processing_time is not None
    assert result.error_message is None
    assert (datetime.now() - result.created_at) < timedelta(seconds=2)

def test_nonexistent_file(ocr_processor, test_dirs):
    """Test processing nonexistent file"""
    input_dir, _ = test_dirs
    nonexistent_file = input_dir / "nonexistent.pdf"
    
    result = ocr_processor(nonexistent_file)
    
    assert result.processing_status == ProcessingStatus.FAILED
    assert result.error_message is not None
    assert result.confidence_score is None
    assert result.raw_text is None

def test_output_file_creation(ocr_processor, test_dirs):
    """Test output file is created with correct content"""
    input_dir, output_dir = test_dirs
    test_file = input_dir / "test_doc.pdf"
    test_file.touch()
    
    result = ocr_processor(test_file)
    
    assert result.output_path.exists()
    assert result.output_path.parent == output_dir
    assert result.output_path.suffix == ".txt"

def test_multiple_documents(ocr_processor, test_dirs):
    """Test processing multiple documents"""
    input_dir, _ = test_dirs
    results = []
    
    # Process multiple documents
    for i in range(3):
        test_file = input_dir / f"test_doc_{i}.pdf"
        test_file.touch()
        results.append(ocr_processor(test_file))
    
    # Check unique document IDs
    doc_ids = [r.document_id for r in results]
    assert len(doc_ids) == len(set(doc_ids))  # All IDs should be unique
    
    # Check all were successful
    assert all(r.processing_status == ProcessingStatus.COMPLETED for r in results)

def test_invalid_input_types(ocr_processor, test_dirs):
    """Test handling of invalid input types"""
    input_dir, _ = test_dirs
    
    # Test with invalid file extension
    invalid_file = input_dir / "test.xyz"
    invalid_file.touch()
    result = ocr_processor(invalid_file)
    assert result.processing_status == ProcessingStatus.COMPLETED  # Mock still processes it
    
    # Test with directory instead of file
    test_dir = input_dir / "test_dir"
    test_dir.mkdir()
    result = ocr_processor(test_dir)
    assert result.processing_status == ProcessingStatus.FAILED

def test_result_immutability(ocr_processor, test_dirs):
    """Test OCRResult immutability"""
    input_dir, _ = test_dirs
    test_file = input_dir / "test_doc.pdf"
    test_file.touch()
    
    result = ocr_processor(test_file)
    
    # Try to modify attributes (should raise error)
    with pytest.raises(Exception):
        result.document_id = "new_id"

def test_processing_time_tracking(ocr_processor, test_dirs):
    """Test processing time is tracked correctly"""
    input_dir, _ = test_dirs
    test_file = input_dir / "test_doc.pdf"
    test_file.touch()
    
    result = ocr_processor(test_file)
    
    assert isinstance(result.processing_time, float)
    assert result.processing_time > 0  # Should take some time
    assert result.processing_time < 2  # Shouldn't take too long

def test_error_handling_edge_cases(ocr_processor, test_dirs):
    """Test various error conditions"""
    input_dir, _ = test_dirs
    
    # Test with empty file
    empty_file = input_dir / "empty.pdf"
    empty_file.touch()
    result = ocr_processor(empty_file)
    assert result.processing_status == ProcessingStatus.COMPLETED
    
    # Test with very large filename - don't actually create file
    long_name_file = input_dir / ("a" * 255 + ".pdf")
    # Skip file creation as it will fail on OS level
    result = ocr_processor(long_name_file)
    assert result.processing_status == ProcessingStatus.FAILED
    assert "name too long" in result.error_message.lower()

def test_post_document_processing(ocr_processor, test_dirs):
    """Test POST document processing with translation"""
    input_dir, _ = test_dirs
    test_file = input_dir / "test_post.pdf"
    test_file.touch()
    
    result = ocr_processor(test_file)
    
    assert isinstance(result, OCRResult)
    assert result.document_id.startswith("DOC_")
    assert result.document_type == DocumentType.POST
    assert result.processing_status == ProcessingStatus.COMPLETED
    assert result.source_language == "de"
    assert result.translated_text is not None
    assert result.translation_confidence is not None
    assert result.raw_text is not None
    assert result.confidence_score is not None
    assert result.error_message is None

def test_document_type_handling(ocr_processor, test_dirs):
    """Test document type detection and enum handling"""
    input_dir, _ = test_dirs
    
    # Test different file types and expected document types
    test_cases = [
        ("invoice.pdf", DocumentType.INVOICE),
        ("receipt.pdf", DocumentType.RECEIPT),
        ("post.pdf", DocumentType.POST),
        ("unknown.pdf", DocumentType.OTHER),
    ]
    
    for filename, expected_type in test_cases:
        test_file = input_dir / filename
        test_file.touch()
        
        result = ocr_processor(test_file)
        
        assert isinstance(result.document_type, DocumentType)
        assert result.document_type in DocumentType
        
        # Check document type matches filename
        if "post" in filename:
            assert result.document_type == DocumentType.POST
            assert result.source_language == "de"
            assert result.translated_text is not None
        elif "invoice" in filename:
            assert result.document_type == DocumentType.INVOICE
            assert result.source_language is None
            assert result.translated_text is None
        
        # Test enum value consistency
        assert result.document_type.value in ["invoice", "receipt", "post", "other"]

def test_document_type_conversion(ocr_processor, test_dirs):
    """Test document type conversion and validation"""
    input_dir, _ = test_dirs
    test_file = input_dir / "test.pdf"
    test_file.touch()
    
    result = ocr_processor(test_file)
    
    # Test conversion to/from string
    doc_type_str = result.document_type.value
    assert DocumentType(doc_type_str) == result.document_type
    
    # Test invalid type conversion
    with pytest.raises(ValueError):
        DocumentType("invalid_type")
