#!/bin/bash

# Load environment variables from .env if it exists
if [ -f "/app/.env" ]; then
    export $(cat /app/.env | xargs)
fi

# Verify key files exist
if [ ! -f "/app/google-translate-key.json" ]; then
    echo "Error: Google Translate key file not found"
    exit 1
fi

# Activate the appropriate virtual environment based on the first argument
if [[ "$1" == "run" && "$2" == "mllm" ]]; then
    echo "Activating MLLM environment..."
    source /opt/venv_mllm/bin/activate
elif [[ "$1" == "run" && "$2" == "tesseract" ]]; then
    echo "Activating Tesseract environment..."
    source /opt/venv_tesseract/bin/activate
fi

# Execute the command
exec python run.py "$@" 