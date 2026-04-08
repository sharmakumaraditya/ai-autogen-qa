import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

DEFAULT_BATCH_SIZE = 5
DEFAULT_MODEL = MODEL_NAME
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
