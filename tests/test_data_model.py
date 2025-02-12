import pytest
from pathlib import Path
from datetime import datetime
from src.data_model import (
    DocumentType,
    ProcessingStatus,
    VisionResult,
    VisionBatch,
    JsonResult,
    JsonBatch
)
from pydantic_core import ValidationError

# Fixture for common test paths
@pytest.fixture
def test_paths():
    return {
        'input': Path('test_input.jpg'),
        'output': Path('test_output.txt'),
        'markdown': Path('test_content.md'),
        'json': Path('test_output.json')
    }

# Test Document Type Enum
def test_document_types():
    assert DocumentType.INVOICE.value == "invoice"
    assert DocumentType.RECEIPT.value == "receipt"
    assert DocumentType.POST.value == "post"
    assert DocumentType.OTHER.value == "other"
    assert len(DocumentType) == 4

# Test Processing Status Enum
def test_processing_status():
    assert ProcessingStatus.PENDING.value == "pending"
    assert ProcessingStatus.PROCESSING.value == "processing"
    assert ProcessingStatus.COMPLETED.value == "completed"
    assert ProcessingStatus.FAILED.value == "failed"
    assert len(ProcessingStatus) == 4

# Test VisionResult
class TestVisionResult:
    def test_minimal_vision_result(self, test_paths):
        """Test creation with minimal required fields"""
        result = VisionResult(
            document_id="test123",
            input_path=test_paths['input'],
            output_path=test_paths['output']
        )
        assert result.document_id == "test123"
        assert result.input_path == test_paths['input']
        assert result.output_path == test_paths['output']
        assert result.document_type == DocumentType.OTHER
        assert result.processing_status == ProcessingStatus.PENDING

    def test_full_vision_result(self, test_paths):
        """Test creation with all fields"""
        result = VisionResult(
            document_id="test123",
            input_path=test_paths['input'],
            output_path=test_paths['output'],
            document_type=DocumentType.INVOICE,
            processing_status=ProcessingStatus.COMPLETED,
            raw_text="Sample text",
            structured_content="# Invoice\n- Item: Value",
            processing_time=1.23,
            created_at=datetime(2024, 1, 1),
            tokens_used=150
        )
        assert result.document_type == DocumentType.INVOICE
        assert result.processing_status == ProcessingStatus.COMPLETED
        assert result.raw_text == "Sample text"
        assert result.structured_content == "# Invoice\n- Item: Value"
        assert result.processing_time == 1.23
        assert result.tokens_used == 150

    def test_vision_result_immutability(self, test_paths):
        """Test that VisionResult is immutable"""
        result = VisionResult(
            document_id="test123",
            input_path=test_paths['input'],
            output_path=test_paths['output']
        )
        with pytest.raises(Exception):
            result.document_id = "new_id"

    def test_vision_result_error_state(self, test_paths):
        """Test error handling in VisionResult"""
        result = VisionResult(
            document_id="test123",
            input_path=test_paths['input'],
            output_path=test_paths['output'],
            processing_status=ProcessingStatus.FAILED,
            error_message="Processing failed"
        )
        assert result.processing_status == ProcessingStatus.FAILED
        assert result.error_message == "Processing failed"

# Test VisionBatch
class TestVisionBatch:
    def test_minimal_vision_batch(self):
        """Test creation with minimal required fields"""
        batch = VisionBatch(
            batch_id="batch123",
            total_documents=5
        )
        assert batch.batch_id == "batch123"
        assert batch.total_documents == 5
        assert len(batch.results) == 0
        assert batch.processed_documents == 0
        assert batch.status == ProcessingStatus.PENDING

    def test_vision_batch_with_results(self, test_paths):
        """Test batch with results"""
        result = VisionResult(
            document_id="test123",
            input_path=test_paths['input'],
            output_path=test_paths['output']
        )
        batch = VisionBatch(
            batch_id="batch123",
            total_documents=1,
            results=[result],
            processed_documents=1,
            status=ProcessingStatus.COMPLETED
        )
        assert len(batch.results) == 1
        assert batch.processed_documents == 1
        assert batch.status == ProcessingStatus.COMPLETED

# Test JsonResult
class TestJsonResult:
    def test_minimal_json_result(self, test_paths):
        """Test creation with minimal required fields"""
        result = JsonResult(
            document_id="test123",
            input_path=test_paths['markdown'],
            output_path=test_paths['json']
        )
        assert result.document_id == "test123"
        assert result.input_path == test_paths['markdown']
        assert result.output_path == test_paths['json']
        assert result.processing_status == ProcessingStatus.PENDING

    def test_full_json_result(self, test_paths):
        """Test creation with all fields"""
        json_content = {"key": "value"}
        result = JsonResult(
            document_id="test123",
            input_path=test_paths['markdown'],
            output_path=test_paths['json'],
            processing_status=ProcessingStatus.COMPLETED,
            json_content=json_content,
            processing_time=0.5,
            document_type=DocumentType.INVOICE
        )
        assert result.json_content == json_content
        assert result.processing_time == 0.5
        assert result.document_type == DocumentType.INVOICE

    def test_json_result_immutability(self, test_paths):
        """Test that JsonResult is immutable"""
        result = JsonResult(
            document_id="test123",
            input_path=test_paths['markdown'],
            output_path=test_paths['json']
        )
        with pytest.raises(Exception):
            result.document_id = "new_id"

# Test JsonBatch
class TestJsonBatch:
    def test_minimal_json_batch(self):
        """Test creation with minimal required fields"""
        batch = JsonBatch(
            batch_id="batch123",
            total_documents=5
        )
        assert batch.batch_id == "batch123"
        assert batch.total_documents == 5
        assert len(batch.results) == 0
        assert batch.processed_documents == 0
        assert batch.status == ProcessingStatus.PENDING

    def test_json_batch_with_results(self, test_paths):
        """Test batch with results"""
        result = JsonResult(
            document_id="test123",
            input_path=test_paths['markdown'],
            output_path=test_paths['json']
        )
        batch = JsonBatch(
            batch_id="batch123",
            total_documents=1,
            results=[result],
            processed_documents=1,
            status=ProcessingStatus.COMPLETED
        )
        assert len(batch.results) == 1
        assert batch.processed_documents == 1
        assert batch.status == ProcessingStatus.COMPLETED

# Test edge cases and validation
def test_edge_cases(test_paths):
    """Test various edge cases and validation"""
    
    # Test empty strings
    with pytest.raises(ValidationError):
        VisionResult(document_id="", input_path=test_paths['input'], output_path=test_paths['output'])
    
    # Test invalid paths (using None instead of string path)
    with pytest.raises(ValidationError):
        VisionResult(document_id="test123", input_path=None, output_path=test_paths['output'])
    
    # Test negative processing time
    with pytest.raises(ValidationError):
        VisionResult(
            document_id="test123",
            input_path=test_paths['input'],
            output_path=test_paths['output'],
            processing_time=-1.0
        )
    
    # Test negative tokens
    with pytest.raises(ValidationError):
        VisionResult(
            document_id="test123",
            input_path=test_paths['input'],
            output_path=test_paths['output'],
            tokens_used=-1
        )
