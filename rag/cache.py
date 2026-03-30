"""
cache.py — Semantic Cache + Retrieval Cache con Redis.

Dos capas de cache independientes:

1. SemanticCache: Compara la pregunta del usuario contra preguntas
   previamente respondidas usando similitud de embeddings. Si encuentra
   una pregunta lo suficientemente similar (threshold), devuelve la
   respuesta cacheada SIN llamar al LLM.

2. RetrievalCache: Cachea los chunks recuperados para un query,
   evitando recalcular embeddings y buscar en el vector store.

Ambas usan Redis como backend y TTL para expiración automática.
"""
import json
import hashlib
import numpy as np
import redis
from typing import Optional

from langchain_core.documents import Document
from rag.config import REDIS_URL, CACHE_TTL, SEMANTIC_CACHE_THRESHOLD
from rag.embeddings import get_embeddings


class SemanticCache:
    """
    Cache semántico: no busca por texto exacto, sino por SIGNIFICADO.

    Flujo:
    1. Usuario pregunta "¿cómo devuelvo un producto?"
    2. Se genera el embedding de esa pregunta
    3. Se compara contra todos los embeddings de preguntas cacheadas
    4. Si alguna tiene similitud >= threshold → cache HIT, devuelve respuesta
    5. Si no → cache MISS, se ejecuta el RAG normal y se guarda el resultado

    Esto significa que "¿cómo devuelvo un producto?" y "¿cuál es la política
    de devoluciones?" pueden ser un HIT si son semánticamente similares.
    """

    PREFIX = "rag:semantic:"

    def __init__(self):
        self._redis = redis.from_url(REDIS_URL, decode_responses=False)
        self._embeddings = get_embeddings()

    def _embed(self, text: str) -> np.ndarray:
        """Genera embedding normalizado para un texto."""
        vec = self._embeddings.embed_query(text)
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similitud coseno entre dos vectores normalizados (= dot product)."""
        return float(np.dot(a, b))

    def get(self, question: str) -> Optional[str]:
        """
        Busca si hay una respuesta cacheada para una pregunta similar.
        Retorna la respuesta o None si no hay cache hit.
        """
        query_vec = self._embed(question)

        # Iterar sobre las entradas del cache
        keys = self._redis.keys(f"{self.PREFIX}*")
        best_score = 0.0
        best_answer = None

        for key in keys:
            data = self._redis.get(key)
            if data is None:
                continue
            entry = json.loads(data)
            cached_vec = np.array(entry["embedding"], dtype=np.float32)
            score = self._cosine_similarity(query_vec, cached_vec)

            if score > best_score:
                best_score = score
                best_answer = entry["answer"]

        if best_score >= SEMANTIC_CACHE_THRESHOLD:
            return best_answer

        return None

    def set(self, question: str, answer: str) -> None:
        """Guarda una pregunta y su respuesta en el cache."""
        vec = self._embed(question)
        key_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
        key = f"{self.PREFIX}{key_hash}"

        entry = {
            "question": question,
            "answer": answer,
            "embedding": vec.tolist(),
        }
        self._redis.setex(key, CACHE_TTL, json.dumps(entry))

    def clear(self) -> int:
        """Limpia todo el cache semántico. Retorna cantidad de keys eliminadas."""
        keys = self._redis.keys(f"{self.PREFIX}*")
        if keys:
            return self._redis.delete(*keys)
        return 0

    def stats(self) -> dict:
        """Estadísticas del cache."""
        keys = self._redis.keys(f"{self.PREFIX}*")
        return {
            "entries": len(keys),
            "ttl_seconds": CACHE_TTL,
            "threshold": SEMANTIC_CACHE_THRESHOLD,
        }


class RetrievalCache:
    """
    Cache de retrieval: cachea los chunks recuperados para un query exacto.

    Más simple que el semántico — usa hash del query como key.
    Evita recalcular el embedding del query y buscar en ChromaDB.
    """

    PREFIX = "rag:retrieval:"

    def __init__(self):
        self._redis = redis.from_url(REDIS_URL, decode_responses=True)

    def _key(self, question: str) -> str:
        h = hashlib.sha256(question.encode()).hexdigest()[:16]
        return f"{self.PREFIX}{h}"

    def get(self, question: str) -> Optional[list[Document]]:
        """Retorna los Documents cacheados o None."""
        data = self._redis.get(self._key(question))
        if data is None:
            return None
        entries = json.loads(data)
        return [
            Document(page_content=e["content"], metadata=e["metadata"])
            for e in entries
        ]

    def set(self, question: str, docs: list[Document]) -> None:
        """Cachea los Documents recuperados."""
        entries = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]
        self._redis.setex(
            self._key(question), CACHE_TTL, json.dumps(entries)
        )

    def clear(self) -> int:
        """Limpia todo el cache de retrieval."""
        keys = self._redis.keys(f"{self.PREFIX}*")
        if keys:
            return self._redis.delete(*keys)
        return 0
