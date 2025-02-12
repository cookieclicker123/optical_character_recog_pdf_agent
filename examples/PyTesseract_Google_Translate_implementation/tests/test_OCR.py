import pytest
import logging
from pathlib import Path
import cv2
import numpy as np

from src.OCR import setup_ocr_pipeline
from src.tools.OCR_tools import (
    load_document,
    preprocess_image,
    perform_ocr,
    detect_layout,
    save_visualization
)
from src.tools.multilingual import (
    detect_and_translate,
    preprocess_financial_text,
    save_multilingual_output
)
from src.data_model import ProcessingStatus, DocumentType, OCRResult

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_dirs():
    """Setup test directories"""
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
    
    # Verify input files exist
    if not input_invoice.exists():
        pytest.skip(f"Test file not found: {input_invoice}")
    if not input_post.exists():
        pytest.skip(f"Test file not found: {input_post}")
    
    # Setup directories
    for dir in [output_invoice_dir, output_post_dir, vis_invoice_dir, vis_post_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'invoice': (input_invoice, output_invoice_dir, vis_invoice_dir),
        'post': (input_post, output_post_dir, vis_post_dir)
    }

def test_document_loading(test_dirs):
    """Test document loading function"""
    input_invoice, _, _ = test_dirs['invoice']
    input_post, _, _ = test_dirs['post']
    
    # Test invoice image loading
    invoice_image = load_document(input_invoice)
    assert isinstance(invoice_image, np.ndarray)
    assert len(invoice_image.shape) == 3  # Should be color image
    
    # Test PDF loading
    post_image = load_document(input_post)
    assert isinstance(post_image, np.ndarray)

def test_image_preprocessing():
    """Test image preprocessing function"""
    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    processed = preprocess_image(test_image)
    
    assert isinstance(processed, np.ndarray)
    assert len(processed.shape) == 2  # Should be grayscale
    assert processed.shape[0] == 400  # 4x resize
    assert processed.shape[1] == 400

def test_ocr_processing(test_dirs):
    """Test OCR processing functions"""
    input_invoice, _, _ = test_dirs['invoice']
    
    # Load and preprocess image
    image = load_document(input_invoice)
    processed = preprocess_image(image)
    
    # Test OCR
    text, confidence = perform_ocr(processed)
    assert isinstance(text, str)
    assert len(text) > 0
    assert 0 <= confidence <= 1

def test_layout_detection(test_dirs):
    """Test layout detection"""
    input_invoice, _, _ = test_dirs['invoice']
    image = load_document(input_invoice)
    
    layout = detect_layout(image)
    assert 'boxes' in layout
    assert 'labels' in layout
    assert 'scores' in layout

def test_multilingual_processing(test_dirs):
    """Test multilingual processing"""
    # Test German text
    german_text = "Dies ist ein Test Dokument"
    original, translated, confidence = detect_and_translate(german_text)
    
    assert original == german_text
    assert translated is not None
    assert confidence is not None
    
    # Test preprocessing
    processed = preprocess_financial_text(german_text)
    assert isinstance(processed, str)
    assert len(processed) > 0

def test_end_to_end_pipeline(test_dirs):
    """Test complete OCR pipeline"""
    for doc_type in ['invoice', 'post']:
        input_file, output_dir, vis_dir = test_dirs[doc_type]
        
        # Setup and run pipeline
        pipeline = setup_ocr_pipeline(input_file, output_dir, vis_dir)
        result = pipeline(input_file)
        
        # Verify result
        assert isinstance(result, OCRResult)
        assert result.processing_status == ProcessingStatus.COMPLETED
        assert result.document_type == (DocumentType.POST if doc_type == 'post' else DocumentType.INVOICE)
        assert result.output_path.exists()
        assert result.raw_text is not None
        
        # Check translation for post documents
        if doc_type == 'post':
            assert result.source_language is not None
            assert result.translated_text is not None
