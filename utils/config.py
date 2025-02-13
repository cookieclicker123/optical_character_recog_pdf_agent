import os

from dotenv import load_dotenv

load_dotenv()

def get_groq_config():
    """Get Groq LLM configuration."""
    return {
        "model_name": os.getenv("GROQ_MODEL_NAME", "llama-3.2-90b-vision-preview"),
        "temperature": 0.1,
        "max_tokens": 8192,
        "provider": "groq",
        "api_key": os.getenv("GROQ_API_KEY"),
        "display_name": "Groq (Llama 3.2 90B Vision Preview)"
    }

def get_openai_config():
    """Get OpenAI Model configuration."""
    return {
        "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
        "temperature": 0.1,
        "max_tokens": 8192,
        "quality": "high",
        "provider": "openai",
        "max_inmage_size": 4096,
        "display_name": "OpenAI (GPT-4o)"
    }