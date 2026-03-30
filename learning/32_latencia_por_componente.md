# Métricas de Latencia por Componente

## Por qué medir por componente

"El RAG es lento" no es accionable. "El LLM tarda 2.1s de los 2.3s totales" sí lo es.

Midiendo por componente identificas exactamente dónde está el cuello de botella y puedes decidir si la optimización vale el esfuerzo.

---

## Distribución típica de latencia en un RAG

```
Total: ~2000ms

LLM:       1700ms ████████████████████ 85%   ← siempre dominante
Retriever:  250ms ███                  12%
Prompt:      30ms                       1%
Format:      20ms                       1%
```

El LLM siempre domina. Pero cuando el retriever supera el 20%, hay un problema — posiblemente el corpus es grande o el modelo de embedding es lento.

---

## Cómo medir

```python
import time

def medir_paso(fn, *args):
    t0 = time.perf_counter()
    resultado = fn(*args)
    ms = (time.perf_counter() - t0) * 1000
    return resultado, ms

docs, ms_retriever = medir_paso(retriever.invoke, query)
context, ms_format = medir_paso(format_docs, docs)
answer, ms_llm = medir_paso(chain_llm.invoke, prompt)
```

Usar `time.perf_counter()` (no `time.time()`) para mayor precisión en mediciones cortas.

---

## Estrategias de optimización por componente

### LLM lento (>1500ms)

| Estrategia | Reducción esperada |
|------------|-------------------|
| Semantic cache (mismo query) | ~100% para cache hits |
| Modelo más rápido (haiku vs sonnet) | 3-5× |
| Reducir tokens de contexto (k más pequeño) | 10-30% |
| Streaming (perceived latency) | No reduce, pero mejora UX |

### Retriever lento (>300ms)

| Estrategia | Reducción esperada |
|------------|-------------------|
| Retrieval cache (Redis) | ~100% para cache hits |
| Reducir chunk count total | 20-50% |
| Índice HNSW en ChromaDB | 2-5× para corpus grandes |
| Migrar a pgvector con índice IVFFlat | Variable |

### Embedding del query lento (parte del retriever)

| Estrategia | Reducción |
|------------|-----------|
| Modelo de embedding más pequeño | 2-3× |
| Cache del embedding | ~100% para queries repetidos |
| GPU para el modelo | 5-10× |

---

## Percentiles vs promedios

El promedio oculta los outliers. En producción, mide P50, P95 y P99:

```python
import statistics

latencias = [1200, 1100, 1300, 4500, 1150, 1200]  # 4500 es un outlier

print(statistics.mean(latencias))    # 1742ms — promedio engañoso
print(statistics.median(latencias))  # 1200ms — P50 real
# P95 real: 4500ms — el 5% de users espera más de 4 segundos
```

Un SLA razonable para RAG: P50 < 2s, P95 < 5s.
