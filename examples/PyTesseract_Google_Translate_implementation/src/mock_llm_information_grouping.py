from pathlib import Path
import time
import json
from typing import Dict, Any
from .data_model import OCRResult, LLMResult, LLMStatus, LLMFn, DocumentType

def setup_llm_pipeline(json_output_dir: Path) -> LLMFn:
    """Factory function that returns an LLM processing function
    
    Args:
        json_output_dir: Directory for JSON output files
        
    Returns:
        Function that takes an OCRResult and returns an LLMResult
    """
    json_output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_document(ocr_result: OCRResult) -> LLMResult:
        """Process OCR result and return structured JSON data"""
        try:
            start_time = time.time()
            
            # Verify OCR result
            if not ocr_result.output_path.exists():
                raise FileNotFoundError(f"OCR text file not found: {ocr_result.output_path}")
            
            if ocr_result.processing_status.value == "failed":
                raise ValueError("Cannot process failed OCR result")
            
            # Read OCR text
            ocr_text = ocr_result.output_path.read_text()
            
            # Generate appropriate mock JSON based on document type
            if ocr_result.document_type == DocumentType.POST:
                mock_json = {
                    "document_identification": {
                        "document_type": "Notice for tax period",
                        "document_id": ocr_result.document_id,
                        "reference_number": f"REF-{ocr_result.document_id}"
                    },
                    "tax_office_information": {
                        "name": "Tax Office Hannover",
                        "address": "PO Box 167, 30001 Hannover",
                        "contact": {
                            "phone": "+49 511 6790-0",
                            "fax": "+49 511 6790-6090"
                        }
                    },
                    "financial_details": {
                        "amounts": {
                            "to_be_settled": 250.03,
                            "already_paid": 0.00,
                            "remaining_balance": 250.03
                        },
                        "dates": {
                            "due_date": "2023-12-31",
                            "notice_date": "2024-01-15"
                        }
                    },
                    "banking_information": {
                        "bank_name": "BBk Hannover",
                        "iban": "DE60 2500 0000 0025 0015 14",
                        "bic": "MARKDEF1250"
                    }
                }
            else:
                # Existing invoice mock JSON
                mock_json = {
                    "invoice_details": {
                        "invoice_number": f"INV-{ocr_result.document_id}",
                        "date": "2024-03-10",
                        "due_date": "2024-04-10"
                    },
                    "amounts": {
                        "subtotal": 250.00,
                        "tax": 50.00,
                        "total": 300.00
                    },
                    "vendor": {
                        "name": "Mock Company",
                        "email": "mock@example.com"
                    },
                    "line_items": [
                        {
                            "description": "Item 1",
                            "quantity": 1,
                            "price": 100.00
                        },
                        {
                            "description": "Item 2",
                            "quantity": 1,
                            "price": 150.00
                        }
                    ]
                }
            
            # Create output JSON file
            json_path = json_output_dir / f"{ocr_result.document_id}.json"
            json_path.write_text(json.dumps(mock_json, indent=2))
            
            processing_time = time.time() - start_time
            
            return LLMResult(
                document_id=ocr_result.document_id,
                input_path=ocr_result.output_path,
                output_path=json_path,
                processing_status=LLMStatus.COMPLETED,
                json_content=mock_json,
                processing_time=processing_time,
                tokens_used=1000,  # Mock token usage
                document_type=ocr_result.document_type
            )
            
        except Exception as e:
            return LLMResult(
                document_id=ocr_result.document_id,
                input_path=ocr_result.output_path,
                output_path=json_output_dir / f"{ocr_result.document_id}_error.json",
                processing_status=LLMStatus.FAILED,
                error_message=str(e),
                document_type=ocr_result.document_type
            )
    
    return process_document 