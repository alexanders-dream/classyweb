# backend/config.py
# (Paste content from your Streamlit config.py here)
# --- Example start (replace with your actual config) ---
import os
from pathlib import Path

USER_HOME = Path.home()
SAVED_MODELS_BASE_DIR = USER_HOME / ".ai_classifier_saved_models"
SAVED_HF_MODELS_BASE_PATH = SAVED_MODELS_BASE_DIR / "hf_models"
# Ensure the directory exists (optional, can be done on demand)
# SAVED_HF_MODELS_BASE_PATH.mkdir(parents=True, exist_ok=True)

# --- HF Training Defaults ---
DEFAULT_VALIDATION_SPLIT = 0.15
DEFAULT_HF_THRESHOLD = 0.5
HF_RULE_COLUMNS = ['Label', 'Keywords', 'Confidence Threshold'] # Add if needed by utils/hf

# --- Hierarchy Definition ---
HIERARCHY_LEVELS = ["Theme", "Category", "Segment", "Subsegment"]

# --- API Defaults ---
DEFAULT_OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
DEFAULT_GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1")

# --- Model Defaults ---
DEFAULT_GROQ_MODEL = "llama3-70b-8192"
DEFAULT_OLLAMA_MODEL = "llama3:latest"

# --- Classification Defaults ---
DEFAULT_LLM_TEMPERATURE = 0.1

# --- UI Defaults (These might eventually move entirely to frontend config) ---
DEFAULT_LLM_SAMPLE_SIZE = 200
MIN_LLM_SAMPLE_SIZE = 50
MAX_LLM_SAMPLE_SIZE = 1000

# --- Provider List ---
SUPPORTED_PROVIDERS = ["Groq", "Ollama"]
# --- End Example ---