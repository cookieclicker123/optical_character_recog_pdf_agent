from enum import Enum
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from pydantic import BaseModel, Field
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

class OCRResult(BaseModel):
    """Result of OCR processing"""
    document_id: str = Field(..., description="Unique identifier for the document")
    input_path: Path = Field(..., description="Path to input document")
    output_path: Path = Field(..., description="Path to OCR text output")
    document_type: DocumentType = Field(default=DocumentType.OTHER)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    raw_text: Optional[str] = Field(default=None, description="Extracted OCR text")
    confidence_score: Optional[float] = Field(default=None, description="Overall OCR confidence")
    processing_time: Optional[float] = Field(default=None, description="Time taken for OCR in seconds")
    created_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    # New fields for translation
    source_language: Optional[str] = Field(default=None, description="Detected source language")
    translated_text: Optional[str] = Field(default=None, description="Translated text if needed")
    translation_confidence: Optional[float] = Field(default=None, description="Translation confidence score")

    class Config:
        arbitrary_types_allowed = True
        frozen = True

class OCRBatch(BaseModel):
    """Batch of documents for processing"""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    results: List[OCRResult] = Field(default_factory=list)
    total_documents: int = Field(..., description="Total number of documents in batch")
    processed_documents: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)


class LLMStatus(Enum):
    """Status of LLM processing"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class LLMResult(BaseModel):
    """Result of LLM processing"""
    document_id: str = Field(..., description="Matching ID from OCR result")
    input_path: Path = Field(..., description="Path to input OCR text file")
    output_path: Path = Field(..., description="Path to output JSON file")
    processing_status: LLMStatus = Field(default=LLMStatus.QUEUED)
    json_content: Optional[Dict[str, Any]] = Field(default=None, description="Structured JSON data")
    llm_reasoning: Optional[str] = Field(default=None, description="LLM's reasoning process")
    processing_time: Optional[float] = Field(default=None, description="Time taken for LLM processing")
    created_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    tokens_used: Optional[int] = Field(default=None, description="Number of tokens used in LLM processing")
    
    class Config:
        arbitrary_types_allowed = True
        frozen = True

class LLMBatch(BaseModel):
    """Batch of documents for LLM processing"""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    results: List[LLMResult] = Field(default_factory=list)
    total_documents: int = Field(..., description="Total number of documents in batch")
    processed_documents: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    status: LLMStatus = Field(default=LLMStatus.QUEUED)



OcrFn = Callable[[Path], OCRResult]
LLMFn = Callable[[OCRResult], LLMResult]
