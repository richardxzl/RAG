# Nodo Retriever — Busca chunks

## Responsabilidades

1. Leer `pregunta` del estado
2. Ejecutar la búsqueda en el vectorstore
3. Escribir `documentos` en el estado
4. Opcionalmente registrar métricas (latencia, scores, fuentes)

---

## Retriever básico

```python
def nodo_retriever(estado: EstadoRAG) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])
    return {
        "documentos": docs,
        "logs": [f"retriever: {len(docs)} docs"],
    }
```

---

## Retriever con scores de similitud

```python
def nodo_retriever_con_scores(estado: EstadoRAG) -> dict:
    vs = get_vectorstore()
    resultados = vs.similarity_search_with_score(estado["pregunta"], k=4)
    docs = []
    for doc, score in resultados:
        doc.metadata["retrieval_score"] = float(score)  # el Grader puede usarlo
        docs.append(doc)
    return {"documentos": docs}
```

Los scores quedan en `doc.metadata["retrieval_score"]` para que el nodo Grader pueda filtrar por umbral.

---

## ChromaDB: distancia vs similitud

ChromaDB por defecto retorna **distancia** (no similitud):
- Distancia 0 = idéntico
- Distancia 1+ = muy diferente

Al contrario de lo que el nombre "similarity_search_with_score" sugiere. Para filtrar, usa un **umbral de distancia** (ej: `score <= 0.7`), no de similitud.

---

## Registrar fuentes

```python
fuentes = list({
    doc.metadata.get("source", "desconocido")
    for doc in docs
})
return {
    "documentos": docs,
    "fuentes": fuentes,
}
```

El set `{}` elimina duplicados cuando varios chunks vienen del mismo archivo.

---

## Métricas de latencia

```python
import time

def nodo_retriever(estado):
    t0 = time.perf_counter()
    docs = retriever.invoke(estado["pregunta"])
    latencia_ms = (time.perf_counter() - t0) * 1000
    return {
        "documentos": docs,
        "latencia_retriever_ms": latencia_ms,
    }
```
