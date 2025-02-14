import pytest
from pathlib import Path
import shutil
from src.json_info_grouping import setup_json_converter
from src.data_model import ProcessingStatus, JsonResult, TranslationResult
import json

@pytest.fixture
def test_dirs():
    """Setup and teardown test directories"""
    base_dir = Path("tests/test_data")
    input_dir = base_dir / "post_ocr_translated_txt"
    output_dir = base_dir / "post_ocr_json"
    fixtures_dir = Path("./tests/fixtures/data/post_ocr_json")
    
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
def json_converter(test_dirs):
    """Create JSON converter function"""
    return setup_json_converter(
        test_dirs['input'],
        test_dirs['output']
    )

def test_successful_conversion(json_converter, test_dirs):
    """Test successful markdown to JSON conversion and fixture creation"""
    input_dir = test_dirs['input']
    fixtures_dir = test_dirs['fixtures']
    
    # Find any txt file in the translated fixtures directory
    translated_fixtures_dir = Path(__file__).parent / "fixtures/data/post_ocr_translated_txt"
    print(f"Looking for txt files in: {translated_fixtures_dir}")
    
    sample_files = list(translated_fixtures_dir.glob("*.txt"))
    if not sample_files:
        pytest.fail(f"No sample txt files found in {translated_fixtures_dir}")
    
    sample_md = sample_files[0]
    print(f"Found sample file: {sample_md}")
    
    # Debug print the content we're sending
    print("\nTranslated content from file:")
    print(sample_md.read_text())
    
    translation_result = TranslationResult(
        document_id=sample_md.stem,
        input_path=sample_md,
        output_path=input_dir / sample_md.name,
        source_language="de",
        target_language="en",
        processing_status=ProcessingStatus.COMPLETED,
        translated_content=sample_md.read_text()
    )
    
    print("\nSending TranslationResult with content:")
    print(translation_result.translated_content)
    
    # Process JSON conversion
    result = json_converter(translation_result)
    
    if result.processing_status == ProcessingStatus.FAILED:
        print("\nConversion failed:")
        print(f"Error: {result.error_message}")
    else:
        print("\nSuccessful conversion:")
        print(json.dumps(result.json_content, indent=2))
    
    # Basic validation
    assert isinstance(result, JsonResult)
    assert result.document_id == translation_result.document_id
    assert result.processing_status == ProcessingStatus.COMPLETED
    assert result.json_content is not None
    assert result.processing_time > 0
    
    # Save fixture
    fixture_path = fixtures_dir / f"{result.document_id}.json"
    fixture_path.write_text(json.dumps(result.json_content, indent=2))
    
    # Verify fixture
    assert fixture_path.exists()
    assert fixture_path.read_text() == json.dumps(result.json_content, indent=2)

def test_error_handling(json_converter, test_dirs):
    """Test JSON conversion errors"""
    # Create a TranslationResult with missing content
    error_translation = TranslationResult(
        document_id="error_doc",
        input_path=Path("nonexistent.txt"),
        output_path=Path("nonexistent_out.txt"),
        source_language="de",
        target_language="en",
        processing_status=ProcessingStatus.COMPLETED,
        translated_content=None  # This should trigger an error
    )
    
    result = json_converter(error_translation)
    
    assert result.processing_status == ProcessingStatus.FAILED
    assert result.error_message is not None 