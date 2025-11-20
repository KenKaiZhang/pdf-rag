import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

# ChromaDB settings
CHROMA_HOST = "localhost"
CHROMA_PORT = "8000"
COLLECTION_NAME = "pdf_tables_collection"

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval parameters
TOP_K_RESULTS = 4