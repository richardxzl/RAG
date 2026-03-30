# Caching en RAG con Redis

## ¿Por qué cache en un RAG?

Cada query a tu RAG tiene costos:

| Operación | Costo | Tiempo |
|-----------|-------|--------|
| Embedding del query | CPU local | ~50ms |
| Búsqueda vectorial (ChromaDB) | CPU + I/O disco | ~20ms |
| Llamada al LLM (Haiku) | Tokens API ($) | 1-3s |

El LLM es el cuello de botella en tiempo Y dinero. Si podemos evitar esa llamada cuando ya tenemos una respuesta válida, ganamos en ambos frentes.

## Las dos capas de cache

### Capa 1: Semantic Cache (la más importante)

**Problema:** El cache exacto (key = pregunta textual) falla con variaciones. "¿Cómo devuelvo un producto?" y "¿Cuál es la política de devoluciones?" son la misma intención pero texto diferente.

**Solución:** En vez de comparar texto, comparamos SIGNIFICADO. Cada pregunta se convierte a embedding y se compara contra las preguntas cacheadas por similitud coseno.

```
"¿Cómo devuelvo un producto?"  →  embedding A
"¿Política de devoluciones?"   →  embedding B

similitud(A, B) = 0.92  →  ≥ 0.85 threshold  →  CACHE HIT
```

**Flujo:**
```
Pregunta nueva
    │
    ▼
Generar embedding
    │
    ▼
Comparar vs embeddings cacheados
    │
    ├── similitud ≥ 0.85 → devolver respuesta cacheada (GRATIS)
    │
    └── similitud < 0.85 → ejecutar RAG normal → cachear resultado
```

**Parámetro crítico: `SEMANTIC_CACHE_THRESHOLD`**

| Valor | Comportamiento | Riesgo |
|-------|---------------|--------|
| 0.95+ | Solo preguntas casi idénticas → pocos hits | Muy conservador, poco ahorro |
| 0.85 | Preguntas con misma intención → buen balance | **Recomendado para empezar** |
| 0.75 | Preguntas vagamente similares → muchos hits | Puede devolver respuestas incorrectas |
| 0.60 | Casi cualquier cosa → hit constante | Peligroso, respuestas irrelevantes |

### Capa 2: Retrieval Cache

Más simple: cachea los chunks recuperados para un query exacto (hash del texto).

```
hash("¿Cómo devuelvo un producto?") → "a3f8c2..."
Redis key: "rag:retrieval:a3f8c2..." → [chunk1, chunk2, chunk3, chunk4]
```

**Ahorra:** Recálculo de embedding del query + búsqueda vectorial. En producción con alta concurrencia, esto importa.

## Redis como backend

### ¿Por qué Redis y no un dict en memoria?

| Aspecto | Dict Python | Redis |
|---------|-------------|-------|
| Persiste entre reinicios | No | Sí |
| Compartido entre procesos | No | Sí |
| TTL automático | No (hay que implementar) | Nativo |
| Escalable | No | Sí (Redis Cluster) |
| Monitoreo | No | `redis-cli monitor`, `INFO` |

Para desarrollo, un dict funciona. Para producción con múltiples workers (FastAPI + Uvicorn), necesitas algo externo como Redis.

### TTL (Time To Live)

Cada entrada expira automáticamente después de N segundos. Esto es CRUCIAL:

- **Sin TTL:** El cache crece infinitamente y las respuestas se vuelven stale si los documentos cambian.
- **TTL muy corto (60s):** Pocos hits, poco ahorro.
- **TTL 1h (3600s):** Buen balance para la mayoría de casos.
- **TTL 24h:** Solo si tus documentos no cambian frecuentemente.

**Regla:** El TTL debe ser menor que la frecuencia de actualización de tus documentos. Si re-ingestas diario, TTL de unas horas. Si los docs son estáticos, TTL de días.

## Cuándo NO usar semantic cache

1. **Chat con historial:** La misma pregunta puede necesitar respuestas diferentes según el contexto de la conversación. Por eso nuestro `chat_fn` solo cachea cuando `chat_history` está vacío.

2. **Documentos que cambian frecuentemente:** Si re-ingestas cada hora, un cache de 1h puede devolver respuestas basadas en docs viejos.

3. **Preguntas que requieren datos en tiempo real:** "¿Cuántas ventas hubo hoy?" no debería cachearse.

## Estructura en Redis

```
rag:semantic:<hash>     → JSON {question, answer, embedding}  (TTL: 3600s)
rag:retrieval:<hash>    → JSON [{content, metadata}, ...]     (TTL: 3600s)
```

El prefijo `rag:` es un namespace que evita colisiones con otras apps que usen el mismo Redis.

## Monitoreo

```bash
# Ver todas las keys de cache
redis-cli KEYS "rag:*"

# Ver cuántas entradas hay
redis-cli KEYS "rag:semantic:*" | wc -l

# Ver el TTL restante de una key
redis-cli TTL "rag:semantic:a3f8c2..."

# Monitorear en tiempo real
redis-cli MONITOR

# Limpiar todo el cache
redis-cli KEYS "rag:*" | xargs redis-cli DEL
```

## Métricas a rastrear en producción

- **Hit rate:** % de preguntas resueltas desde cache. Si es < 10%, tu threshold es muy alto o tus preguntas son muy variadas.
- **Latencia con/sin cache:** Cache hit debería ser < 100ms vs 1-3s sin cache.
- **Memoria Redis:** `INFO memory`. Cada entrada semántica pesa ~2-4KB (embedding + texto).
- **Tokens ahorrados:** Cada cache hit ahorra ~500-2000 tokens de API.

## Evolución futura

1. **Redis con vector search (RediSearch):** En vez de iterar keys y comparar embeddings en Python, Redis busca por similitud nativo. Mucho más rápido con miles de entradas.
2. **Cache warming:** Pre-cachear las preguntas más frecuentes al iniciar.
3. **Cache invalidation por documento:** Cuando re-ingestas un documento, invalidar solo las entradas del cache que usaron chunks de ese documento.
