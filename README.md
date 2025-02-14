# OCR on unstructured Documents

## Project Vision: Goals + Purpose

vision.

## Design principles



## Rough Repo Structure



## Development roadmap



## Open Source Everything



## Setup And pipeline run

Test MLLM implementation and you will see in fixtures/data/post_docs/post_ocr_txt your txt documents with the extracted information.

```bash
git clone ...
cd ocr_on_unstructured_documents

brew install poppler
# Run single pipeline for tesseract
python run.py run tesseract --input-dir ./input --output-dir ./output
# Run single pipeline for mllm tests
python run.py run mllm --mode test --input-dir ./input --output-dir ./output
# Run both pipelines for comparison
python run.py run compare --input-dir ./input --output-dir ./output

touch .env
OPENAI_API_KEY='your-api-key-here'
GROQ_API_KEY='your-api-key-here'
GOOGLE_APPLICATION_CREDENTIALS=./google-translate-key.json
```