import pytest
from pathlib import Path
from src.OCR import setup_ocr_pipeline
from src.data_model import ProcessingStatus, DocumentType, OCRResult

@pytest.fixture
def test_dirs():
    """Setup test directories"""
    # Define paths within fixtures
    fixtures_dir = Path("./tests/fixtures")
    input_file = fixtures_dir / "data/invoice.png"
    output_dir = fixtures_dir / "ocr_txt_files"
    vis_dir = fixtures_dir / "ocr_visualization"
    
    # Verify input file exists
    if not input_file.exists():
        pytest.skip(f"Test file not found: {input_file}")
    
    # Setup output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    return input_file, output_dir, vis_dir

@pytest.fixture
def ocr_processor(test_dirs):
    """Create OCR processor function"""
    input_file, output_dir, vis_dir = test_dirs
    return setup_ocr_pipeline(input_file, output_dir, vis_dir)

def test_ocr_file_creation_and_content(ocr_processor, test_dirs):
    """Verify OCR output file creation and content in fixtures directory"""
    input_file, output_dir, vis_dir = test_dirs
    
    # Process the file
    result = ocr_processor(input_file)
    
    # Verify file creation and content
    txt_file = result.output_path
    assert txt_file.exists(), f"OCR output file not created: {txt_file}"
    assert txt_file.suffix == ".txt"
    assert txt_file.parent == output_dir
    assert txt_file.stem.startswith("DOC_")
    assert len(txt_file.stem) == 12  # "DOC_" + 8 chars
    
    # Verify content
    ocr_text = txt_file.read_text()
    assert ocr_text == result.raw_text
    assert len(ocr_text) > 0, "OCR output is empty"
    
    # Verify visualization
    vis_file = vis_dir / f"{result.document_id}_visualization.png"
    assert vis_file.exists(), "Visualization not created"
    
    # Verify OCR result object
    assert isinstance(result, OCRResult)
    assert result.processing_status == ProcessingStatus.COMPLETED
    assert result.document_type == DocumentType.INVOICE 