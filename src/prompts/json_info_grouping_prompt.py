MARKDOWN_TO_JSON_PROMPT = """Convert this markdown document to structured JSON.
Group information into these categories:

{
    "document_info": {
        "id_number": "",
        "tax_number": "",
        "form_number": "",
        "document_type": "",
        "date": ""
    },
    "contact": {
        "address": "",
        "phone": "",
        "fax": "",
        "website": ""
    },
    "tax_details": {
        "period": "",
        "reference_numbers": [],
        "amounts": {
            "total": "",
            "paid": "",
            "outstanding": ""
        }
    },
    "banking": {
        "accounts": [{
            "bank": "",
            "iban": "",
            "bic": ""
        }]
    }
}

Rules:
1. Keep all numbers exactly as written
2. Group multi-line text appropriately
3. Convert dates to ISO format
4. Preserve all reference numbers
5. Include all amounts with currency"""