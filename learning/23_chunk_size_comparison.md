# Comparar Chunk Sizes — Test automatizado

## Por qué importa el chunk size

El chunk size es el parámetro que más impacta la calidad del RAG. No hay un valor universalmente óptimo — depende del dominio, del modelo de embedding, y del tipo de queries.

| chunk_size pequeño (500) | chunk_size grande (1500) |
|--------------------------|--------------------------|
| ✅ Búsqueda más precisa | ✅ Más contexto para el LLM |
| ✅ Menos solapamiento entre chunks recuperados | ✅ Menos fragmentación de ideas |
| ❌ Puede truncar ideas incompletas | ❌ Búsqueda más difusa (más ruido) |
| ❌ El LLM recibe poco contexto | ❌ Más tokens = más costo |

---

## Qué medir

Un buen test de chunk size compara estas métricas para un conjunto de queries de evaluación:

| Métrica | Qué indica |
|---------|-----------|
| `num_chunks` | Cuántos chunks genera la configuración |
| `avg_chars` | Tamaño promedio de los chunks |
| `solapamiento` | Redundancia entre los K chunks recuperados (↓ mejor) |
| `tokens_contexto` | Costo de tokens por query (↓ menor costo) |
| `latencia` | Tiempo de retrieval (suele ser similar entre configuraciones) |

---

## Cómo funciona el test

```python
# Para cada chunk_size en [500, 1000, 1500]:
#   1. Re-chunkear los documentos originales
#   2. Indexar en un vector store temporal (en memoria)
#   3. Para cada query de evaluación:
#      - Recuperar K chunks
#      - Calcular solapamiento entre ellos
#      - Calcular tokens de contexto
#   4. Promediar métricas

for chunk_size in [500, 1000, 1500]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2),  # 20% de overlap siempre
    )
    chunks = splitter.split_documents(docs_originales)
    vs_temp = Chroma(embedding_function=embeddings)
    vs_temp.add_documents(chunks)
    # ... evaluar queries
```

---

## Relación chunk_overlap / chunk_size

El overlap debe escalar con el chunk size. Un ratio del 15-20% es razonable:

| chunk_size | overlap recomendado |
|-----------|---------------------|
| 500 | 75-100 |
| 1000 | 150-200 |
| 1500 | 225-300 |

Un overlap muy alto aumenta redundancia. Muy bajo, y el contexto se pierde en los bordes.

---

## Limitación de este test

Este test mide **métricas proxy** (solapamiento, tokens), no calidad real de las respuestas. Para medir calidad necesitas:

1. Un dataset de evaluación con preguntas + respuestas esperadas (Módulo 5.1)
2. Métricas como faithfulness y answer relevance (Módulo 5.2)
3. Herramientas como RAGAS (Módulo 5.3)

Los resultados de este test son un punto de partida, no una respuesta definitiva.

---

## Patrón de iteración recomendado

```
1. Empieza con chunk_size=1000, overlap=200 (el default de este proyecto)
2. Evalúa con preguntas representativas del dominio
3. Si el LLM dice "no tengo información" frecuentemente → chunk más pequeño (mejor precisión)
4. Si las respuestas son superficiales o incompletas → chunk más grande (más contexto)
5. Si ves mucha redundancia en los chunks recuperados → chunk más pequeño + MMR
6. Itera hasta encontrar el punto de equilibrio para tu caso de uso específico
```
