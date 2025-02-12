import pytest
from pathlib import Path
import shutil
from datetime import datetime, timedelta
from src.mock_OCR import setup_vision_pipeline
from src.data_model import ProcessingStatus, DocumentType, VisionResult

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
def vision_processor(test_dirs):
    """Create Vision processor function"""
    input_dir, output_dir = test_dirs
    return setup_vision_pipeline(input_dir, output_dir)

def test_factory_function_setup(test_dirs):
    """Test factory function creates directories and returns VisionFn"""
    input_dir, output_dir = test_dirs
    processor = setup_vision_pipeline(input_dir, output_dir)
    
    assert input_dir.exists()
    assert output_dir.exists()
    assert callable(processor)

def test_successful_processing(vision_processor, test_dirs):
    """Test successful document processing"""
    input_dir, _ = test_dirs
    test_file = input_dir / "test_doc.pdf"
    test_file.touch()
    
    result = vision_processor(test_file)
    
    assert isinstance(result, VisionResult)
    assert result.document_id.startswith("DOC_")
    assert result.processing_status == ProcessingStatus.COMPLETED
    assert result.input_path == test_file
    assert result.raw_text is not None
    assert result.structured_content is not None
    assert result.processing_time is not None
    assert result.tokens_used is not None
    assert result.error_message is None
    assert (datetime.now() - result.created_at) < timedelta(seconds=2)

def test_nonexistent_file(vision_processor, test_dirs):
    """Test processing nonexistent file"""
    input_dir, _ = test_dirs
    nonexistent_file = input_dir / "nonexistent.pdf"
    
    result = vision_processor(nonexistent_file)
    
    assert result.processing_status == ProcessingStatus.FAILED
    assert result.error_message is not None
    assert result.raw_text is None
    assert result.structured_content is None

def test_output_file_creation(vision_processor, test_dirs):
    """Test output file is created with correct content"""
    input_dir, output_dir = test_dirs
    test_file = input_dir / "test_doc.pdf"
    test_file.touch()
    
    result = vision_processor(test_file)
    
    assert result.output_path.exists()
    assert result.output_path.parent == output_dir
    assert result.output_path.suffix == ".txt"
    
    # Check that the output contains markdown structure
    content = result.output_path.read_text()
    assert "#" in content  # Should have markdown headers
    assert "-" in content  # Should have markdown list items

def test_multiple_documents(vision_processor, test_dirs):
    """Test processing multiple documents"""
    input_dir, _ = test_dirs
    results = []
    
    # Process multiple documents
    for i in range(3):
        test_file = input_dir / f"test_doc_{i}.pdf"
        test_file.touch()
        results.append(vision_processor(test_file))
    
    # Check unique document IDs
    doc_ids = [r.document_id for r in results]
    assert len(doc_ids) == len(set(doc_ids))  # All IDs should be unique
    
    # Check all were successful
    assert all(r.processing_status == ProcessingStatus.COMPLETED for r in results)

def test_invalid_input_types(vision_processor, test_dirs):
    """Test handling of invalid input types"""
    input_dir, _ = test_dirs
    
    # Test with directory instead of file
    test_dir = input_dir / "test_dir"
    test_dir.mkdir()
    result = vision_processor(test_dir)
    assert result.processing_status == ProcessingStatus.FAILED

def test_result_immutability(vision_processor, test_dirs):
    """Test VisionResult immutability"""
    input_dir, _ = test_dirs
    test_file = input_dir / "test_doc.pdf"
    test_file.touch()
    
    result = vision_processor(test_file)
    
    # Try to modify attributes (should raise error)
    with pytest.raises(Exception):
        result.document_id = "new_id"

def test_processing_time_tracking(vision_processor, test_dirs):
    """Test processing time is tracked correctly"""
    input_dir, _ = test_dirs
    test_file = input_dir / "test_doc.pdf"
    test_file.touch()
    
    result = vision_processor(test_file)
    
    assert isinstance(result.processing_time, float)
    assert result.processing_time > 0  # Should take some time
    assert result.processing_time < 2  # Shouldn't take too long

def test_error_handling_edge_cases(vision_processor, test_dirs):
    """Test various error conditions"""
    input_dir, _ = test_dirs
    
    # Test with empty file
    empty_file = input_dir / "empty.pdf"
    empty_file.touch()
    result = vision_processor(empty_file)
    assert result.processing_status == ProcessingStatus.COMPLETED
    
    # Test with very large filename
    long_name_file = input_dir / ("a" * 255 + ".pdf")
    result = vision_processor(long_name_file)
    assert result.processing_status == ProcessingStatus.FAILED
    assert "name too long" in result.error_message.lower()

def test_document_type_handling(vision_processor, test_dirs):
    """Test document type detection and structured content format"""
    input_dir, _ = test_dirs
    
    # Test different document types
    test_cases = [
        ("post.pdf", DocumentType.POST),
        ("invoice.pdf", DocumentType.INVOICE),
    ]
    
    for filename, expected_type in test_cases:
        test_file = input_dir / filename
        test_file.touch()
        
        result = vision_processor(test_file)
        
        assert result.document_type == expected_type
        assert isinstance(result.structured_content, str)
        assert result.structured_content.startswith("#")  # Markdown header
        
        if expected_type == DocumentType.POST:
            assert "Letter" in result.structured_content
        else:
            assert "Invoice" in result.structured_content
