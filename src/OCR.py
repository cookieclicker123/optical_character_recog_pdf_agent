from pathlib import Path
import time
import uuid
from .data_model import VisionResult, ProcessingStatus, DocumentType, VisionFn
from .tools.vision_tool import setup_vision_tool
from .prompts.post_prompts import EXTRACT_PROMPT
from utils.config import get_openai_config

def setup_vision_pipeline(input_dir: Path, output_dir: Path) -> VisionFn:
    """Factory function that returns a Vision processing function
    
    Args:
        input_dir: Directory for input documents
        output_dir: Directory for output
        
    Returns:
        Function that takes a Path and returns a VisionResult
    """
    # Setup directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup vision tool
    vision_tool = setup_vision_tool(get_openai_config())
    
    def process_document(input_path: Path) -> VisionResult:
        """Process a single document and return Vision result"""
        try:
            # Validate input
            if not input_path.exists():
                raise FileNotFoundError(f"File not found: {input_path}")
            
            if input_path.is_dir():
                raise IsADirectoryError(f"Expected file, got directory: {input_path}")
                
            if len(input_path.name) > 240:  # Leave room for path
                raise ValueError("Filename too long")
            
            # Start timing
            start_time = time.time()
            
            # Generate unique document ID
            doc_id = f"DOC_{uuid.uuid4().hex[:8]}"
            
            # Create output path
            output_path = output_dir / f"{doc_id}.txt"
            
            # Extract content using vision tool
            extract_result = vision_tool(input_path, EXTRACT_PROMPT)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Write the output file
            output_path.write_text(extract_result.markdown)
            
            return VisionResult(
                document_id=doc_id,
                input_path=input_path,
                output_path=output_path,
                document_type=DocumentType.POST,  # Always POST for now
                processing_status=ProcessingStatus.COMPLETED,
                raw_text=extract_result.response,
                structured_content=extract_result.markdown,
                processing_time=processing_time,
                tokens_used=extract_result.tokens_used,
                vision_tool_calls=[extract_result]
            )
            
        except Exception as e:
            return VisionResult(
                document_id=f"DOC_{uuid.uuid4().hex[:8]}",
                input_path=input_path,
                output_path=output_dir / "error.txt",
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e),
                document_type=DocumentType.OTHER
            )
    
    return process_document 