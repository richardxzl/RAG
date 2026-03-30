# Batch — Procesar múltiples preguntas en paralelo

## El problema que resuelve

Un loop de `.invoke()` es serial: la segunda pregunta no empieza hasta que la primera termina.

```python
# Serial — cada llamada espera a la anterior
respuestas = []
for pregunta in preguntas:
    respuesta = chain.invoke({"question": pregunta})  # bloquea aquí
    respuestas.append(respuesta)

# Tiempo total = suma de todos los tiempos individuales
# 5 preguntas × 2s cada una = 10s mínimo
```

`.batch()` ejecuta todas las invocaciones en paralelo usando un `ThreadPoolExecutor`:

```python
# Paralelo — todas las llamadas se solapan en tiempo
inputs = [{"question": q} for q in preguntas]
respuestas = chain.batch(inputs)

# Tiempo total ≈ el de la invocación más lenta
# 5 preguntas, la más lenta tarda 2.5s = ~2.5s total
```

---

## `.batch()` vs múltiples `.invoke()` en loop

La diferencia no es magia: es concurrencia a nivel de **threads**, no async verdadero.

Internamente, LangChain hace esto:

```python
from concurrent.futures import ThreadPoolExecutor

def batch(self, inputs, config=None, max_concurrency=None):
    workers = max_concurrency or len(inputs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(self.invoke, inp, cfg)
                   for inp, cfg in zip(inputs, configs)]
        return [f.result() for f in futures]
```

Mientras el thread de la pregunta 1 espera la respuesta de la API, los threads de las preguntas 2, 3, 4 ya hicieron sus llamadas. El I/O de red se solapa.

**Cuándo hay ganancia real**: cuando el bottleneck es I/O de red (llamadas al LLM, APIs externas). El tiempo de espera de la red se amortiza entre todos los threads.

**Cuándo NO hay ganancia real**: si el bottleneck es CPU local. El retriever con ChromaDB y embeddings locales (`all-MiniLM-L6-v2`) corre en CPU. Si retrieve tarda 800ms y el LLM tarda 200ms, el cuello de botella es local y threading no ayuda — el GIL de Python además limita el paralelismo CPU real.

---

## `max_concurrency`

Si no se especifica, `max_concurrency` es igual a `len(inputs)`. Con 20 preguntas, abre 20 threads simultáneos y hace 20 llamadas a la API al mismo tiempo.

El problema: las APIs de LLM tienen rate limits.

| Límite | Descripción |
|--------|-------------|
| RPM (Requests Per Minute) | Máximo de requests en una ventana de 60s |
| TPM (Tokens Per Minute) | Máximo de tokens procesados en 60s |

Si superas el límite, la API responde con HTTP 429 y la invocación falla. Por defecto, si una invocación del batch falla, la excepción se propaga y el batch entero falla.

```python
# Sin límite — puede disparar 429 con datasets grandes
respuestas = chain.batch(inputs)

# Con límite — máximo 2 llamadas simultáneas
respuestas = chain.batch(inputs, max_concurrency=2)
```

Regla práctica para dimensionar `max_concurrency`:

```
max_concurrency = floor(RPM_límite / 60 × tiempo_promedio_por_llamada_en_segundos)
```

Ejemplo: 60 RPM, llamadas de ~2s → `max_concurrency = floor(60/60 × 2) = 2`.

---

## `RunnableConfig` por item

`.batch()` acepta un segundo argumento: una lista de `RunnableConfig`, una por input.

```python
from langchain_core.runnables import RunnableConfig

inputs = [{"question": q} for q in preguntas]

configs = [
    RunnableConfig(
        tags=["pregunta-1", "experimento-A"],
        metadata={"usuario_id": "u001", "pregunta_index": 1},
        run_name="rag-pregunta-1",
    ),
    RunnableConfig(
        tags=["pregunta-2", "experimento-A"],
        metadata={"usuario_id": "u001", "pregunta_index": 2},
        run_name="rag-pregunta-2",
    ),
]

# La lista de configs debe tener la misma longitud que inputs
respuestas = chain.batch(inputs, configs)
```

Campos útiles de `RunnableConfig`:

| Campo | Tipo | Para qué sirve |
|-------|------|----------------|
| `tags` | `list[str]` | Filtrar traces en LangSmith |
| `metadata` | `dict` | Datos arbitrarios adjuntos al trace |
| `run_name` | `str` | Nombre legible en LangSmith |
| `callbacks` | `list` | Handlers custom (logging, métricas) |
| `max_concurrency` | `int` | Override por invocación (raro) |

La config no modifica el output — es información para el sistema de observabilidad y los callbacks.

---

## `.abatch()` — versión async

`.abatch()` es la versión async de `.batch()`. Usa `asyncio` en lugar de threads.

```python
import asyncio

async def procesar():
    inputs = [{"question": q} for q in preguntas]
    respuestas = await chain.abatch(inputs)
    return respuestas

# En un script síncrono:
respuestas = asyncio.run(procesar())
```

**Cuándo usar `.abatch()` en lugar de `.batch()`**:

- Entorno ya async: FastAPI, notebooks con `await`, servidores asyncio
- Quieres no bloquear el event loop mientras se procesan las preguntas
- Tienes muchos batches concurrentes y quieres que cooperen entre sí

**Cuándo `.batch()` con threads es suficiente**:

- Scripts de línea de comandos (`python 08_batch.py`)
- Pipelines de datos síncronos
- No hay event loop activo

La diferencia en rendimiento es marginal para batches pequeños. La diferencia importante es el modelo de concurrencia: threads vs corrutinas.

---

## Cuándo batch NO ayuda

El mito: "batch siempre es más rápido".

La realidad: batch es más rápido solo si el bottleneck es I/O de red.

**Caso donde batch no gana**:

```
Pipeline: retriever_local (800ms) → LLM_API (200ms)

Batch de 5 preguntas:
  Thread 1: [retriever 800ms][LLM 200ms] = 1000ms
  Thread 2: [retriever 800ms]...     ← compite por CPU con Thread 1
  Thread 3: [retriever 800ms]...     ← idem

El GIL de Python impide que los threads Python-puro corran en paralelo real.
SentenceTransformer sí libera el GIL durante la inferencia de numpy,
pero la contención de CPU sigue siendo un factor.

Resultado: el tiempo total no es ~1000ms, puede ser 2000ms-3000ms
porque los threads se pisan entre sí en la fase de retrieval.
```

**Cuándo batch SÍ gana mucho**:

- El retriever es una API externa (Pinecone, Weaviate Cloud): I/O puro
- El LLM es remoto (Anthropic, OpenAI): I/O puro
- El ratio LLM/retriever es alto (el LLM domina el tiempo)

---

## Caso de uso ideal

```python
# Evaluar un dataset de preguntas con respuestas esperadas
dataset = [
    {"question": "¿X?", "expected": "La respuesta es X"},
    {"question": "¿Y?", "expected": "La respuesta es Y"},
    # ... 100 entradas más
]

inputs = [{"question": item["question"]} for item in dataset]
respuestas = chain.batch(inputs, max_concurrency=5)

# Ahora comparar respuestas con expected
for item, respuesta in zip(dataset, respuestas):
    score = evaluate(respuesta, item["expected"])
    print(f"{score:.2f} | {item['question'][:50]}")
```

Sin batch, evaluar 100 preguntas a 2s cada una = 200s mínimo. Con batch y `max_concurrency=5`, el tiempo teórico es ~40s.

Otros casos ideales:
- Indexar/resumir múltiples documentos en paralelo
- Generar embeddings de una colección de textos
- Procesar columnas de un dataframe donde cada fila es independiente

---

## Error handling en batch

Por defecto, si una invocación del batch lanza una excepción, **el batch entero falla**:

```python
preguntas = ["pregunta válida", "", "otra válida"]  # la segunda romperá
try:
    respuestas = chain.batch([{"question": q} for q in preguntas])
except Exception as e:
    print(f"El batch falló: {e}")
    # No hay respuestas parciales disponibles aquí
```

Para tolerancia a fallos, usa `return_exceptions=True`:

```python
respuestas = chain.batch(
    inputs,
    config=None,
    return_exceptions=True,  # las excepciones se devuelven como valores
)

for i, resultado in enumerate(respuestas):
    if isinstance(resultado, Exception):
        console.print(f"[red]Pregunta {i} falló: {resultado}[/]")
    else:
        console.print(f"[green]Pregunta {i}: {resultado[:60]}[/]")
```

Con `return_exceptions=True`, el batch siempre termina. Las invocaciones exitosas devuelven su resultado; las fallidas devuelven el objeto excepción. El llamador decide qué hacer con cada caso.

---

## Tradeoffs

| Aspecto | Detalle |
|---------|---------|
| **Memoria** | Todos los resultados se acumulan en RAM antes de retornar. Un batch de 1000 respuestas de 500 tokens cada una puede ocupar varios MB. Para datasets grandes, considera procesar en sub-batches. |
| **Rate limits** | Sin `max_concurrency`, un batch grande puede superar los límites de la API y disparar errores 429. Dimensionar con cuidado. |
| **Errores parciales** | Por defecto, un fallo cancela todo. Usar `return_exceptions=True` para robustez. |
| **Bottleneck local** | Si el retriever es CPU-bound (embeddings locales), el beneficio del paralelismo se reduce por el GIL. |
| **Orden garantizado** | `.batch()` preserva el orden: `respuestas[i]` corresponde a `inputs[i]`. |
| **No streaming** | `.batch()` no hace streaming. Si necesitas respuestas incrementales, `.stream()` o `.astream_events()` son los caminos. |

---

## Implementación en este proyecto

Ver [08_batch.py](../../08_batch.py) para la demo completa con los tres escenarios.

El pipeline construido tiene esta forma:

```
inputs = [{"question": q1}, {"question": q2}, ...]
          │
          ▼
  .batch(inputs, max_concurrency=N)
          │
          ├── Thread 1: RunnablePassthrough.assign(context=retriever+format_docs)
          │             → QUERY_PROMPT → LLM → StrOutputParser
          │
          ├── Thread 2: RunnablePassthrough.assign(...)
          │             → QUERY_PROMPT → LLM → StrOutputParser
          │
          └── Thread N: ...
                        │
                        ▼
          [respuesta_1, respuesta_2, ..., respuesta_N]
          (orden preservado, en RAM, retornado todo junto)
```

---

## Regla de oro

> Usa `.batch()` cuando tengas N inputs independientes y el LLM (o API externa) sea el bottleneck.
> Agrega `max_concurrency` si tu API tiene rate limits.
> Usa `return_exceptions=True` si necesitas tolerancia a fallos en el batch.
