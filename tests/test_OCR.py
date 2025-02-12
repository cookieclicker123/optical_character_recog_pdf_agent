import pytest
import logging
from pathlib import Path
from src.OCR import setup_ocr_pipeline
from src.data_model import ProcessingStatus, DocumentType, OCRResult

# Setup file logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # This will still show logs in console too
    ]
)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_dirs():
    """Setup test directories"""
    # Define paths within fixtures
    fixtures_dir = Path("./tests/fixtures")
    input_invoice = fixtures_dir / "data/invoice.png"
    input_post = fixtures_dir / "data/post_1.pdf"
    output_invoice_dir = fixtures_dir / "ocr_txt_files"
    output_post_dir = fixtures_dir / "ocr_post_txt_files"
    vis_invoice_dir = fixtures_dir / "ocr_visualization"
    vis_post_dir = fixtures_dir / "ocr_post_visualization"
    
    # Debug logging
    logger.debug(f"Checking invoice path: {input_invoice}, exists: {input_invoice.exists()}")
    logger.debug(f"Checking post path: {input_post}, exists: {input_post.exists()}")
    logger.debug(f"Current working directory: {Path.cwd()}")
    logger.debug(f"Absolute invoice path: {input_invoice.absolute()}")
    logger.debug(f"Absolute post path: {input_post.absolute()}")
    
    # Verify input files exist
    if not input_invoice.exists():
        logger.error(f"Invoice file not found: {input_invoice}")
        pytest.skip(f"Test file not found: {input_invoice}")
    if not input_post.exists():
        logger.error(f"Post file not found: {input_post}")
        pytest.skip(f"Test file not found: {input_post}")
    
    # Setup output directories
    output_invoice_dir.mkdir(parents=True, exist_ok=True)
    output_post_dir.mkdir(parents=True, exist_ok=True)
    vis_invoice_dir.mkdir(parents=True, exist_ok=True)
    vis_post_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'invoice': (input_invoice, output_invoice_dir, vis_invoice_dir),
        'post': (input_post, output_post_dir, vis_post_dir)
    }

@pytest.fixture
def ocr_processor(test_dirs):
    """Create OCR processor function"""
    def get_processor(doc_type='invoice'):
        input_file, output_dir, vis_dir = test_dirs[doc_type]
        return setup_ocr_pipeline(input_file, output_dir, vis_dir)
    return get_processor

def test_ocr_file_creation_and_content(ocr_processor, test_dirs):
    """Verify OCR output file creation and content for invoice"""
    input_file, output_dir, vis_dir = test_dirs['invoice']
    processor = ocr_processor('invoice')  # Specify invoice processor
    result = processor(input_file)
    
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

def test_post_ocr_processing(ocr_processor, test_dirs):
    """Verify OCR processing for POST document with translation"""
    input_file, output_dir, vis_dir = test_dirs['post']
    processor = ocr_processor('post')  # Specify post processor
    result = processor(input_file)
    
    # Verify file creation and content
    txt_file = result.output_path
    assert txt_file.exists(), f"OCR output file not created: {txt_file}"
    assert txt_file.suffix == ".txt"
    assert txt_file.parent == output_dir
    assert txt_file.stem.startswith("DOC_")
    
    # Verify content
    ocr_text = txt_file.read_text()
    assert len(ocr_text) > 0, "OCR output is empty"
    
    # Verify translation fields
    assert result.source_language == "de", "Source language should be German"
    assert result.translated_text is not None, "Translation should be present"
    assert result.translation_confidence is not None, "Translation confidence should be present"
    
    # Verify visualization
    vis_file = vis_dir / f"{result.document_id}_visualization.png"
    assert vis_file.exists(), "Visualization not created"
    
    # Verify OCR result object
    assert isinstance(result, OCRResult)
    assert result.processing_status == ProcessingStatus.COMPLETED
    assert result.document_type == DocumentType.POST
    
    # Verify both original and translated text are present
    assert "Original Text:" in ocr_text
    assert "Translated Text:" in ocr_text

def test_google_translate_setup():
    """Verify Google Cloud Translation setup and credentials"""
    from google.cloud import translate_v2 as translate
    from dotenv import load_dotenv
    import os
    
    # Load .env file
    load_dotenv()
    
    # Check if credentials path is set
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    assert creds_path is not None, "GOOGLE_APPLICATION_CREDENTIALS not set in .env"
    assert Path(creds_path).exists(), f"Credentials file not found at {creds_path}"
    
    try:
        # Try to create a client and do a simple translation
        client = translate.Client()
        result = client.translate('Hallo Welt', target_language='en')
        
        # Verify translation worked
        assert result['translatedText'] == 'Hello World', \
            f"Unexpected translation: {result['translatedText']}"
        print("\n✅ Google Translate credentials working!")
        print(f"Test translation: 'Hallo Welt' -> '{result['translatedText']}'")
        
    except Exception as e:
        print("\n❌ Google Translate setup failed!")
        raise Exception(f"Translation failed: {str(e)}") 