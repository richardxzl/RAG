# Contextual Compression — Filtrar chunks irrelevantes post-retrieval

## El problema del ruido en los chunks

El retriever recupera chunks completos aunque solo una parte sea relevante para el query:

```
Query: "¿Qué es LCEL?"

Chunk recuperado (1000 chars):
  "...historia de LangChain, que comenzó en 2022...
   LCEL (LangChain Expression Language) es el sistema de composición moderno...
   ...también hay otras herramientas como LangSmith y LangServe...
   ...el equipo de LangChain tiene sede en San Francisco..."
```

El LLM recibe 1000 chars. Solo 100 responden la pregunta. El resto es ruido que consume tokens y puede confundir la respuesta.

`ContextualCompressionRetriever` es un wrapper que aplica un **compressor** a los docs después de recuperarlos.

---

## Estrategia 1: LLMChainExtractor

Llama al LLM con cada chunk + el query y le pide que extraiga solo la parte relevante:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

docs = retriever.invoke("¿Qué es LCEL?")
# docs contienen solo el extracto relevante, no el chunk completo
```

**Costo**: 1 llamada LLM extra por cada chunk recuperado. Con k=4, son 4 llamadas extra.

---

## Estrategia 2: EmbeddingsFilter

Calcula la similitud coseno entre el query y cada chunk. Si la similitud es menor que el threshold, descarta el chunk:

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter

compressor = EmbeddingsFilter(
    embeddings=get_embeddings(),
    similarity_threshold=0.76,
)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
```

**Sin llamadas LLM extra** — solo vectorización (mucho más barato).

---

## Comparativa

| | LLMChainExtractor | EmbeddingsFilter |
|---|---|---|
| **Mecanismo** | LLM extrae el fragmento relevante | Filtra chunks por similitud |
| **Llamadas LLM extra** | k (una por chunk) | 0 |
| **Latencia** | Alta | Baja |
| **Calidad** | Alta (entiende el contexto) | Media (solo similitud de embedding) |
| **Reduce longitud de chunks** | ✅ Extrae fragmentos | ❌ Elimina chunks enteros |

---

## Combinar ambas estrategias

Puedes encadenar compressors con `DocumentCompressorPipeline`:

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

pipeline = DocumentCompressorPipeline(
    transformers=[
        EmbeddingsFilter(embeddings=embs, similarity_threshold=0.75),  # 1. Filtrar rápido
        LLMChainExtractor.from_llm(llm),                               # 2. Extraer preciso
    ]
)

retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever,
)
```

EmbeddingsFilter descarta los chunks claramente irrelevantes (sin costo LLM). LLMChainExtractor solo opera sobre los que pasaron el filtro → menos llamadas LLM.

---

## Cuándo usar

| Situación | Usar |
|-----------|------|
| Chunks largos con mucho contenido mezclado | ✅ |
| Contexto de tokens limitado (quieres mandar menos) | ✅ |
| Latencia crítica | ❌ LLMChainExtractor agrega latencia |
| Corpus bien chunkeado y limpio | ❌ Overhead sin ganancia |

---

## Threshold en EmbeddingsFilter

El valor depende del modelo de embeddings y del dominio. Para `all-MiniLM-L6-v2`:
- `0.70` → conservador, pasan más chunks
- `0.76` → balance
- `0.82` → agresivo, solo los más relevantes pasan

Empieza en `0.76` y ajusta viendo cuántos chunks pasan en promedio.
