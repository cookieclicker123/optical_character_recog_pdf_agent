from enum import Enum
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any, Annotated
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, constr
from datetime import datetime

class DocumentType(Enum):
    """Types of documents we can process"""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    POST = "post"
    OTHER = "other"

class ProcessingStatus(Enum):
    """Status of document processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VisionResult(BaseModel):
    """Result of MLLM vision processing"""
    document_id: str = Field(
        ..., 
        min_length=1,
        description="Unique identifier for the document"
    )
    input_path: Path = Field(..., description="Path to input document")
    output_path: Path = Field(..., description="Path to extracted text output")
    document_type: DocumentType = Field(default=DocumentType.OTHER)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    raw_text: Optional[str] = Field(default=None, description="Raw extracted text from image")
    structured_content: Optional[str] = Field(default=None, description="Structured markdown content")
    processing_time: Optional[PositiveFloat] = Field(default=None, description="Time taken for processing in seconds")
    created_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    tokens_used: Optional[PositiveInt] = Field(default=None, description="Number of tokens used in MLLM processing")
    
    class Config:
        arbitrary_types_allowed = True
        frozen = True

class VisionBatch(BaseModel):
    """Batch of documents for processing"""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    results: List[VisionResult] = Field(default_factory=list)
    total_documents: int = Field(..., description="Total number of documents in batch")
    processed_documents: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)

class JsonResult(BaseModel):
    """Result of markdown to JSON conversion"""
    document_id: str = Field(..., description="Matching ID from vision result")
    input_path: Path = Field(..., description="Path to markdown content file")
    output_path: Path = Field(..., description="Path to output JSON file")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    json_content: Optional[Dict[str, Any]] = Field(default=None, description="Structured JSON data")
    processing_time: Optional[float] = Field(default=None, description="Time taken for conversion")
    created_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    document_type: DocumentType = Field(default=DocumentType.OTHER)
    
    class Config:
        arbitrary_types_allowed = True
        frozen = True

class JsonBatch(BaseModel):
    """Batch of JSON conversions"""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    results: List[JsonResult] = Field(default_factory=list)
    total_documents: int = Field(..., description="Total number of documents in batch")
    processed_documents: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)

# Type hints for processing functions
VisionFn = Callable[[Path], VisionResult]
JsonFn = Callable[[VisionResult], JsonResult]
