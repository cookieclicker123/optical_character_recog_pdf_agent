import pytest
from pathlib import Path
import shutil
from src.OCR import setup_vision_pipeline
from src.data_model import ProcessingStatus, DocumentType, VisionResult

@pytest.fixture
def test_dirs():
    """Setup and teardown test directories"""
    base_dir = Path("tests/test_data")
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    fixtures_dir = Path("./tests/fixtures/data/post_ocr_txt")
    
    # Setup
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    yield {
        'input': input_dir,
        'output': output_dir,
        'fixtures': fixtures_dir
    }
    
    # Teardown - keep fixtures, clean up test data
    shutil.rmtree(base_dir)

@pytest.fixture
def vision_processor(test_dirs):
    """Create vision processor function"""
    return setup_vision_pipeline(test_dirs['input'], test_dirs['output'])

def test_successful_processing(vision_processor, test_dirs):
    """Test successful document processing and fixture creation"""
    # Setup test document
    input_dir = test_dirs['input']
    fixtures_dir = test_dirs['fixtures']
    test_file = input_dir / "test_letter.pdf"
    
    # Copy sample PDF to input
    sample_pdf = Path("./tests/fixtures/data/post_docs/post_0001.pdf")
    shutil.copy(sample_pdf, test_file)
    
    # Process document
    result = vision_processor(test_file)
    
    # Basic validation
    assert isinstance(result, VisionResult)
    assert result.document_id.startswith("DOC_")
    assert result.processing_status == ProcessingStatus.COMPLETED
    assert result.document_type == DocumentType.POST
    assert result.raw_text is not None
    assert result.structured_content is not None
    assert result.vision_tool_calls is not None
    assert len(result.vision_tool_calls) == 1
    
    # Validate tool call
    tool_call = result.vision_tool_calls[0]
    assert tool_call.error_message is None
    assert tool_call.tokens_used > 0
    assert tool_call.processing_time > 0
    
    # Save fixture
    fixture_path = fixtures_dir / f"{result.document_id}.txt"
    fixture_path.write_text(result.structured_content)
    
    # Verify fixture
    assert fixture_path.exists()
    assert fixture_path.read_text() == result.structured_content

def test_error_handling(vision_processor, test_dirs):
    """Test processing errors"""
    input_dir = test_dirs['input']
    nonexistent_file = input_dir / "nonexistent.pdf"
    
    result = vision_processor(nonexistent_file)
    
    assert result.processing_status == ProcessingStatus.FAILED
    assert result.error_message is not None
    assert result.vision_tool_calls == []