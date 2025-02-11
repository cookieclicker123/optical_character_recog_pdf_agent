import os
from dotenv import load_dotenv

load_dotenv()

def get_groq_config():
    """Get Groq LLM configuration."""
    return {
        "model_name": os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b"),
        "temperature": 0.7,
        "max_tokens": 4096,
        "provider": "groq",
        "api_key": os.getenv("GROQ_API_KEY"),
        "display_name": "Groq (DeepSeek R1 Distill LLaMA 70B)"
    }

