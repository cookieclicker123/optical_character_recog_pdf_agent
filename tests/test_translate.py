import pytest
from pathlib import Path
import shutil
from src.translate import setup_translation_pipeline
from src.data_model import ProcessingStatus, TranslationResult

@pytest.fixture
def test_dirs():
    """Setup and teardown test directories"""
    base_dir = Path("tests/test_data")
    input_dir = base_dir / "post_ocr_txt"
    output_dir = base_dir / "post_ocr_translated_txt"
    fixtures_dir = Path("./tests/fixtures/data/post_ocr_translated_txt")
    
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
def translation_processor(test_dirs):
    """Create translation processor function"""
    return setup_translation_pipeline(
        test_dirs['input'],
        test_dirs['output']
    )

def test_successful_translation(translation_processor, test_dirs):
    """Test successful document translation and fixture creation"""
    # Setup test document
    input_dir = test_dirs['input']
    fixtures_dir = test_dirs['fixtures']
    
    # Find any txt file in the OCR fixtures directory
    ocr_fixtures_dir = Path("./tests/fixtures/data/post_ocr_txt")
    sample_files = list(ocr_fixtures_dir.glob("*.txt"))
    if not sample_files:
        pytest.fail("No sample txt files found in OCR fixtures directory")
    
    sample_md = sample_files[0]  # Take the first txt file found
    test_file = input_dir / sample_md.name
    shutil.copy(sample_md, test_file)
    
    # Process translation
    result = translation_processor(test_file)
    
    # Basic validation
    assert isinstance(result, TranslationResult)
    assert result.document_id == sample_md.stem  # Use the actual filename stem
    assert result.processing_status == ProcessingStatus.COMPLETED
    assert result.translated_content is not None
    assert result.processing_time > 0
    
    # Save fixture
    fixture_path = fixtures_dir / test_file.name
    fixture_path.write_text(result.translated_content)
    
    # Verify fixture
    assert fixture_path.exists()
    assert fixture_path.read_text() == result.translated_content

def test_error_handling(translation_processor, test_dirs):
    """Test translation errors"""
    input_dir = test_dirs['input']
    nonexistent_file = input_dir / "nonexistent.txt"
    
    result = translation_processor(nonexistent_file)
    
    assert result.processing_status == ProcessingStatus.FAILED
    assert result.error_message is not None 