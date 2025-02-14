from pathlib import Path
import time
import uuid
import json
from groq import Groq
from .data_model import JsonResult, ProcessingStatus, JsonFn, TranslationResult
from utils.config import get_groq_config
from .prompts.json_info_grouping_prompt import MARKDOWN_TO_JSON_PROMPT

def setup_json_converter(
    input_dir: Path,
    output_dir: Path,
    config: dict = None
) -> JsonFn:
    """Factory function that returns a JSON conversion function"""
    if config is None:
        config = get_groq_config()
    
    # Setup directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Groq client
    client = Groq(api_key=config['api_key'])
    
    def convert_to_json(translation_result: TranslationResult) -> JsonResult:
        """Convert a translated markdown to JSON"""
        try:
            # Start timing
            start_time = time.time()
            
            # Create output path
            output_path = output_dir / f"{translation_result.document_id}.json"
            
            # Read translated content
            content = translation_result.translated_content
            if not content:
                raise ValueError("No translated content available")
            
            print("\nInput content to LLM:")
            print(content)
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "system",
                    "content": MARKDOWN_TO_JSON_PROMPT
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Call the model
            response = client.chat.completions.create(
                model=config['model_name'],
                messages=messages,
                temperature=0.1,
                max_tokens=config['max_tokens']
            )
            
            # Extract raw response and clean it
            raw_response = response.choices[0].message.content
            print("\nRaw LLM Response:")
            print(raw_response)
            
            # Clean the response of markdown code fence
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            try:
                # Try to parse JSON
                json_content = json.loads(cleaned_response)
                print("\nSuccessfully parsed JSON:")
                print(json.dumps(json_content, indent=2))
            except json.JSONDecodeError as je:
                print(f"\nJSON parsing failed: {str(je)}")
                raise
            
            # Write JSON output
            output_path.write_text(json.dumps(json_content, indent=2))
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return JsonResult(
                document_id=translation_result.document_id,
                input_path=translation_result.output_path,
                output_path=output_path,
                processing_status=ProcessingStatus.COMPLETED,
                json_content=json_content,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"\nError during conversion: {str(e)}")
            return JsonResult(
                document_id=translation_result.document_id,
                input_path=translation_result.output_path,
                output_path=output_dir / "error.json",
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    return convert_to_json 