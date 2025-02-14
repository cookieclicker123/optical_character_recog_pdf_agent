from enum import Enum
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
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

class VisionTool(BaseModel):
    """Result from MLLM vision tool call"""
    prompt: str = Field(..., description="Prompt sent to vision model")
    response: str = Field(..., description="Raw response from vision model")
    markdown: str = Field(..., description="Structured markdown output")
    tokens_used: int = Field(..., ge=0, description="Tokens used in this tool call")
    processing_time: float = Field(..., ge=0.0, description="Time taken for this tool call")
    error_message: Optional[str] = Field(default=None, description="Error if tool call failed")
    
    class Config:
        frozen = True

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
    vision_tool_calls: List[VisionTool] = Field(
        default_factory=list,
        description="List of vision tool calls made during processing"
    )
    
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

class TranslationResult(BaseModel):
    """Result of document translation"""
    document_id: str = Field(..., description="Matching ID from vision result")
    input_path: Path = Field(..., description="Path to source markdown file")
    output_path: Path = Field(..., description="Path to translated output file")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    translated_content: Optional[str] = Field(default=None, description="Translated markdown content")
    processing_time: Optional[float] = Field(default=None, description="Time taken for translation")
    created_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = Field(default=None, description="Error message if translation failed")
    
    class Config:
        arbitrary_types_allowed = True
        frozen = True

class TranslationBatch(BaseModel):
    """Batch of translations"""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    results: List[TranslationResult] = Field(default_factory=list)
    total_documents: int = Field(..., description="Total number of documents in batch")
    processed_documents: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)

# Type hints for processing functions
VisionFn = Callable[[Path], VisionResult]
JsonFn = Callable[[TranslationResult], JsonResult]
VisionToolFn = Callable[[Path, str], VisionTool]
TranslationFn = Callable[[Path], TranslationResult]
