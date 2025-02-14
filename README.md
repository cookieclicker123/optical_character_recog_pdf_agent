# OCR on Unstructured Documents

## Project Vision: Goals + Purpose

Automated extraction and structuring of information from multilingual documents using:
- GPT-4 Vision for OCR and initial structuring
- Google Cloud Translate for translation
- Groq LLM for JSON conversion

## Design Principles

- Clean separation of concerns
- Strong type hints and validation
- Comprehensive error handling
- Test-driven development
- Modular pipeline architecture

## Repository Structure

```
.
├── src/                    # Source code
│   ├── data_model.py      # Pydantic models
│   ├── vision_tool.py     # GPT-4 Vision implementation
│   ├── translate.py       # Google Translate implementation
│   └── json_info_grouping.py  # Groq JSON conversion
├── tests/                  # Test suite
├── input/                  # Input documents
├── output/                 # Generated outputs
├── docker-compose.yml      # Docker configuration
├── Dockerfile             # Container definition
├── requirements_*.txt     # Dependencies
└── run.py                 # CLI runner
```

## Setup And Pipeline Run

### Initial Setup

1. Clone the repository
```bash
git clone ...
cd ocr_on_unstructured_documents
```

### Prerequisites

2. API Keys (add to `.env` in root directory):
```bash
OPENAI_API_KEY=your-openai-key
GROQ_API_KEY=your-groq-key
```

3. Google Cloud Translate credentials:
- Save as `google-translate-key.json` in root directory

### Using Docker (Recommended for Production)

1. Build and run:
```bash
# Build container
docker compose build

# Run full pipeline
docker compose up

# Or run specific mode
docker compose run ocr run mllm --mode test --input-dir /app/input --output-dir /app/output
```

### Using Local Environment (Development)

1. System dependencies:
```bash
# macOS
brew install poppler

# Ubuntu
sudo apt-get install poppler-utils
```

2. Python setup:
```bash
# Create and setup environments
python run.py setup all

# Or setup specific environment
python run.py setup mllm
python run.py setup tesseract
```

3. Run pipeline:
```bash
# Run MLLM pipeline tests
python run.py run mllm --mode test --input-dir ./input --output-dir ./output

# Run Tesseract pipeline
python run.py run tesseract --input-dir ./input --output-dir ./output

# Run comparison of both
python run.py run compare --input-dir ./input --output-dir ./output
```

## Pipeline Flow

1. Document Processing:
   - PDF → Images
   - Images → Markdown (DE)
   - Markdown (DE) → Markdown (EN)
   - Markdown (EN) → JSON

2. Output Structure:
```
output/
├── post_docs/                  # Processed PDFs
├── post_ocr_txt/              # German markdown
├── post_ocr_translated_txt/   # English markdown
└── post_ocr_json/            # Structured JSON
```

## Development Roadmap

- [x] PDF processing
- [x] GPT-4 Vision OCR
- [x] Translation pipeline
- [x] JSON structuring
- [x] Docker containerization
- [ ] API endpoint
- [ ] Web interface

## Open Source

This project is open source under the MIT license. Contributions welcome!

