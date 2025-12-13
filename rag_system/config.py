"""
Configuration settings for the RAG system
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
ARTICLES_DIR = BASE_DIR / "Articles"
DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Document processing settings
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # words overlap between chunks
MIN_CHUNK_SIZE = 100  # minimum words for a valid chunk

# Embedding model settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# FAISS index settings
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
CHUNKS_METADATA_PATH = DATA_DIR / "chunks_metadata.json"
SUGGESTED_QUESTIONS_PATH = DATA_DIR / "preguntas_sugeridas.txt"

# Generation model settings
# Using smaller model for efficiency - can upgrade to gpt-neo-2.7B if needed
GENERATION_MODEL = "EleutherAI/gpt-neo-125M"  
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95

# Retrieval settings
TOP_K_RESULTS = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
