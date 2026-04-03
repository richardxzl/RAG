"""
config.py — Configuración centralizada del proyecto RAG.

Single source of truth para todos los parámetros. Los scripts
y módulos importan de aquí en vez de definir sus propias constantes.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────
DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"

# ── ChromaDB ───────────────────────────────────────────────
COLLECTION_NAME = "mi_knowledge_base"

# ── Embeddings ─────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384

# ── Chunking ──────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval ─────────────────────────────────────────────
RETRIEVAL_K = 4
RETRIEVAL_SEARCH_TYPE = "similarity"

# ── LLM ───────────────────────────────────────────────────
# Formato: "proveedor/modelo"  →  anthropic/..., openai/..., ollama/..., etc.
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))  # ignorado en Ollama

# ── Memory ────────────────────────────────────────────────
MEMORY_WINDOW = 6  # Últimos N mensajes en el chat

# ── Redis Cache ───────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 3600  # 1 hora en segundos
SEMANTIC_CACHE_THRESHOLD = 0.85  # Similitud mínima para cache hit (0-1)
