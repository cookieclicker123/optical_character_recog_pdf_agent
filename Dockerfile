# Use Python 3.11 slim as base
FROM python:3.11-slim

# Install system dependencies including poppler
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements_mllm.txt requirements_tesseract.txt ./

# Copy all application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY utils/ ./utils/
COPY run.py ./

# Environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/google-translate-key.json

# Create virtual environments
RUN python -m venv /opt/venv_mllm && \
    /opt/venv_mllm/bin/pip install -r requirements_mllm.txt && \
    python -m venv /opt/venv_tesseract && \
    /opt/venv_tesseract/bin/pip install -r requirements_tesseract.txt

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--help"] 