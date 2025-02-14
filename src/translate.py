from pathlib import Path
import time
import uuid
from google.cloud import translate_v2 as translate
from .data_model import (
    TranslationResult, 
    ProcessingStatus, 
    TranslationFn
)

def setup_translation_pipeline(
    input_dir: Path,
    output_dir: Path,
    source_lang: str = "de",
    target_lang: str = "en"
) -> TranslationFn:
    """Factory function that returns a Translation processing function
    
    Args:
        input_dir: Directory containing markdown files to translate
        output_dir: Directory for translated output
        source_lang: Source language code (default: 'de')
        target_lang: Target language code (default: 'en')
        
    Returns:
        Function that takes a Path and returns a TranslationResult
    """
    # Setup directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize translation client
    translate_client = translate.Client()
    
    def process_translation(input_path: Path) -> TranslationResult:
        """Translate a single markdown file"""
        try:
            # Validate input
            if not input_path.exists():
                raise FileNotFoundError(f"File not found: {input_path}")
            
            if input_path.is_dir():
                raise IsADirectoryError(f"Expected file, got directory: {input_path}")
            
            # Start timing
            start_time = time.time()
            
            # Extract document ID from filename
            doc_id = input_path.stem
            
            # Create output path
            output_path = output_dir / input_path.name
            
            # Read markdown content
            content = input_path.read_text()
            
            # Translate content while preserving markdown
            result = translate_client.translate(
                content,
                target_language=target_lang,
                source_language=source_lang,
                format_="text"  # Preserve markdown formatting
            )
            
            translated_content = result["translatedText"]
            
            # Write translated content
            output_path.write_text(translated_content)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return TranslationResult(
                document_id=doc_id,
                input_path=input_path,
                output_path=output_path,
                source_language=source_lang,
                target_language=target_lang,
                processing_status=ProcessingStatus.COMPLETED,
                translated_content=translated_content,
                processing_time=processing_time
            )
            
        except Exception as e:
            return TranslationResult(
                document_id=doc_id if 'doc_id' in locals() else f"DOC_{uuid.uuid4().hex[:8]}",
                input_path=input_path,
                output_path=output_dir / "error.txt",
                source_language=source_lang,
                target_language=target_lang,
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    return process_translation 