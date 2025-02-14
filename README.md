# OCR on Unstructured Documents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![GPT-4V](https://img.shields.io/badge/GPT--4V-Vision-green.svg)](https://openai.com/gpt-4)
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-blue.svg)](https://github.com/tesseract-ocr/tesseract)
[![Google Translate](https://img.shields.io/badge/Google-Translate-blue.svg)](https://cloud.google.com/translate)
[![Groq](https://img.shields.io/badge/Groq-LLM-orange.svg)](https://groq.com)
[![JSON](https://img.shields.io/badge/JSON-Structured-lightgrey.svg)](https://www.json.org)
[![PDF](https://img.shields.io/badge/PDF-Processing-red.svg)](https://poppler.freedesktop.org)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-green.svg)](https://docs.pydantic.dev/latest/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com)

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
- Robust data model suitable for furthur development

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

3. **Google Cloud Translate credentials**:
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

