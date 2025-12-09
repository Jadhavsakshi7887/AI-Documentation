"""
Configuration file for RAG Bot application
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# File Upload Configuration
TEMP_UPLOADS_DIR = BASE_DIR / "temp_uploads"
TEMP_UPLOADS_DIR.mkdir(exist_ok=True)

# Vector Store Configuration
PERSIST_FOLDER = os.getenv("PERSIST_FOLDER", str(BASE_DIR / "chroma_vectorstore"))
VECTORSTORE_PERSIST_DIR = Path(PERSIST_FOLDER)
VECTORSTORE_PERSIST_DIR.mkdir(exist_ok=True)

# Embedding Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_FOLDER = os.getenv("EMBEDDING_CACHE_FOLDER", str(BASE_DIR / "huggingface_models"))

# RAG Configuration
RAG_API_KEY = os.getenv("RAG_API_KEY", "sk-voidai-zAFjyYrdhOfQwZahZFXpJm4U3DC4wni8BgbWyxi99B05kiIxkCitl3LjE6z09DBsCNuJnUE0JAGwcZleA4BGJOVaAnHWp6TuHQRQIRQK5EEnuHCRBTGh0hpSDT3kib_LkY19HQ")
RAG_API_URL = os.getenv("RAG_API_URL", "https://api.voidai.app/v1/chat/completions")
RAG_MODEL = os.getenv("RAG_MODEL", "gpt-4o")

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Streamlit Configuration
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))

