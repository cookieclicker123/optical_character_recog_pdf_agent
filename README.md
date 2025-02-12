# OCR on unstructured Documents

## Project Vision: Goals + Purpose

vision.

## Design principles



## Rough Repo Structure



## Development roadmap



## Open Source Everything



## Initial Setup

```bash
git clone ...
cd ocr_on_unstructured_documents
python3 -m venv .venv
pip install -r requirements.txt
touch .env
GROQ_API_KEY='your-api-key-here'
```

## Run The Tests

```bash
pytest tests/test_mock_OCR.py
pytest tests/test_OCR.py
pytest tests/test_mock_llm_information_grouping.py
pytest tests/test_llm_information_grouping.py
pytest tests/test_groq_basic.py
```