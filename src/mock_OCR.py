from pathlib import Path
import time
import uuid
from .data_model import VisionResult, ProcessingStatus, DocumentType, VisionFn

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
            
            # Simulate processing time
            time.sleep(0.5)
            
            # Generate unique document ID
            doc_id = f"DOC_{uuid.uuid4().hex[:8]}"
            
            # Determine document type from filename
            is_post = "post" in input_path.name.lower()
            doc_type = DocumentType.POST if is_post else DocumentType.INVOICE
            
            # Create output path and mock text based on document type
            output_path = output_dir / f"{doc_id}.txt"
            
            if doc_type == DocumentType.POST:
                raw_text = f"""
                Sehr geehrte Damen und Herren,
                
                Ich schreibe Ihnen bezüglich #{doc_id}.
                Das Wetter ist heute schön.
                
                Mit freundlichen Grüßen,
                Hans Schmidt
                """
                structured_content = f"""# Letter
- Type: Business Letter
- Reference: {doc_id}
- Language: German
- Sender: Hans Schmidt
- Content: Weather discussion"""
            else:
                raw_text = f"""
                Invoice #{doc_id}
                Date: 2024-03-10
                
                Item 1: $100.00
                Item 2: $150.00
                
                Total: $250.00
                """
                structured_content = f"""# Invoice
- Document ID: {doc_id}
- Date: 2024-03-10
- Items:
  - Item 1: $100.00
  - Item 2: $150.00
- Total: $250.00"""
            
            # Write the output file
            output_path.write_text(structured_content)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return VisionResult(
                document_id=doc_id,
                input_path=input_path,
                output_path=output_path,
                document_type=doc_type,
                processing_status=ProcessingStatus.COMPLETED,
                raw_text=raw_text,
                structured_content=structured_content,
                processing_time=processing_time,
                tokens_used=150
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
