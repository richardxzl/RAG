"""
embeddings.py — Factory de embeddings.

Centraliza la creación del modelo de embeddings para garantizar
que ingesta, retrieval y cache usen SIEMPRE el mismo modelo.
"""
from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import EMBEDDING_MODEL

_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Singleton: carga el modelo una sola vez y lo reutiliza."""
    global _instance
    if _instance is None:
        _instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _instance
