# Streaming — Respuestas token por token en tiempo real

## El problema que resuelve

Sin streaming, el flujo es: usuario espera → LLM genera toda la respuesta → respuesta aparece de golpe. Para respuestas largas, eso puede ser 5–15 segundos de silencio.

Con streaming, el LLM envía cada token a medida que lo genera. La latencia percibida cae drásticamente porque el usuario ve texto desde el primer segundo.

```python
# Sin streaming: espera total = tiempo de generación completo
answer = chain.invoke({"question": pregunta, "docs": docs})
print(answer)  # aparece todo de una vez

# Con streaming: primer token aparece en ~300ms
for chunk in chain.stream({"question": pregunta, "docs": docs}):
    print(chunk, end="", flush=True)
```

---

## `.stream()` vs `.astream()` — sync vs async

LangChain expone ambas variantes de la misma API:

| Método | Modelo de ejecución | Cuándo usarlo |
|--------|---------------------|---------------|
| `.stream()` | Síncrono — genera un iterador Python estándar | Scripts, CLIs, contextos sin event loop |
| `.astream()` | Asíncrono — genera un async generator | FastAPI, aplicaciones con `asyncio`, concurrencia real |

```python
# Sync — uso directo con for
for chunk in chain.stream(input):
    print(chunk, end="", flush=True)

# Async — requiere async def y await
async def stream_response():
    async for chunk in chain.astream(input):
        print(chunk, end="", flush=True)
```

La diferencia NO es solo sintáctica. `.stream()` bloquea el thread mientras espera tokens del LLM. `.astream()` libera el event loop entre tokens, permitiendo atender otras solicitudes concurrentemente.

**En una API web con 100 usuarios simultáneos**: `.stream()` necesitaría 100 threads. `.astream()` maneja todos en un event loop.

---

## Qué retorna `.stream()`: chunk types

El tipo de cada chunk depende de si hay `StrOutputParser` al final del pipeline:

### Con `StrOutputParser` (lo más común)

```python
chain = prompt | llm | StrOutputParser()

for chunk in chain.stream(input):
    print(type(chunk))  # → <class 'str'>
    print(repr(chunk))  # → 'La ' | 'respuesta' | ' es...'
```

Cada chunk es un `str`. `StrOutputParser` extrae `.content` de cada `AIMessageChunk` y lo retorna como string simple.

### Sin `StrOutputParser`

```python
chain = prompt | llm  # sin parser

for chunk in chain.stream(input):
    print(type(chunk))        # → <class 'AIMessageChunk'>
    print(chunk.content)      # → 'La ' | 'respuesta' | ' es...'
    print(chunk.response_metadata)  # → {} en chunks intermedios, info al final
```

`AIMessageChunk` es más verboso pero da acceso a metadata. El chunk final tiene `response_metadata` con `stop_reason`, `usage`, etc.

---

## Qué componentes streamean de verdad vs cuáles no

Este es el punto conceptual más importante. En un pipeline RAG, NO todos los pasos pueden streamear:

```
pregunta
    │
    ▼
Retriever.invoke()          ← BLOQUEA. Debe terminar ANTES de que el LLM empiece.
    │                          No puede streamear porque necesita todos los docs
    │                          para construir el contexto del prompt.
    ▼
RunnablePassthrough.assign()  ← BLOQUEA. Espera que todos sus assign() terminen
    │                            antes de pasar el dict al siguiente paso.
    ▼
ChatPromptTemplate            ← BLOQUEA. Construye el prompt completo antes de enviarlo.
    │
    ▼
ChatAnthropic (LLM)           ← STREAMEA. Emite tokens a medida que los genera.
    │                            Este es el único componente del pipeline RAG
    │                            que produce streaming real.
    ▼
StrOutputParser               ← PASS-THROUGH. Transforma cada AIMessageChunk en str
                                 sin buffering. El streaming del LLM se preserva.
```

**Consecuencia práctica**: cuando llamas `.stream()` sobre el pipeline completo, los primeros pasos se ejecutan síncronamente (el retriever, el formateo, la construcción del prompt). El streaming solo empieza cuando el LLM recibe su input. Todo el tiempo previo es latencia fija.

### Por qué el retriever no puede streamear

El retriever hace una búsqueda vectorial y retorna una lista de `Document`. El prompt necesita TODOS los documentos antes de poder construirse — no puede empezar a renderizarse con "algunos" docs y completarse después. Es una dependencia de datos completa, no incremental.

Analogía: no puedes preparar la receta antes de tener todos los ingredientes.

---

## `.stream_events()` — La API de eventos de bajo nivel

`.stream_events()` expone el grafo de ejecución del pipeline como un stream de eventos estructurados. Es más detallado que `.stream()` y útil para observabilidad.

```python
for event in chain.stream_events(input, version="v2"):
    print(event)
```

Cada evento es un dict con esta estructura:

```python
{
    "event":   "on_llm_stream",        # tipo del evento
    "name":    "ChatAnthropic",        # nombre del componente
    "run_id":  "abc-123",              # ID único de esta ejecución
    "tags":    [],                     # tags heredados del pipeline
    "metadata": {},                    # metadata del pipeline
    "data": {                          # payload específico por tipo de evento
        "chunk": AIMessageChunk(...)   # en on_llm_stream
        # o "input": ... / "output": ... en otros eventos
    }
}
```

### Eventos principales

| Evento | Cuándo ocurre | `data` contiene |
|--------|---------------|-----------------|
| `on_chain_start` | Un Runnable empieza | `input` |
| `on_chain_stream` | Un Runnable emite chunk intermedio | `chunk` |
| `on_chain_end` | Un Runnable termina | `output` |
| `on_llm_start` | LLM recibe su prompt | `input` con los mensajes |
| `on_llm_stream` | LLM emite un token | `chunk` como `AIMessageChunk` |
| `on_llm_end` | LLM termina de generar | `output` con `LLMResult` |
| `on_retriever_start` | Retriever empieza búsqueda | `query` |
| `on_retriever_end` | Retriever retorna docs | `documents` |

### Filtrar el ruido

Un pipeline RAG simple puede emitir 50+ eventos. Para logs útiles, filtrar:

```python
EVENTOS_RELEVANTES = {"on_llm_start", "on_llm_stream", "on_llm_end", "on_retriever_end"}

for event in chain.stream_events(input, version="v2"):
    if event["event"] in EVENTOS_RELEVANTES:
        procesar(event)
```

### `version="v2"` es obligatorio

La `v1` está deprecated. Siempre pasar `version="v2"`. La diferencia principal es que v2 incluye los eventos de todos los sub-runnables anidados, no solo el nivel superior.

---

## Integración con HTTP SSE (Server-Sent Events)

En producción, streaming va a un cliente web via SSE — una conexión HTTP de larga duración donde el servidor envía líneas `data: ...\n\n`.

**Concepto sin implementar**: en FastAPI, la integración sería:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_answer(question: str):
    async def generate():
        async for chunk in chain.astream({"question": question, "docs": docs}):
            yield f"data: {chunk}\n\n"   # formato SSE estándar
        yield "data: [DONE]\n\n"         # señal de fin

    return StreamingResponse(generate(), media_type="text/event-stream")
```

El cliente JavaScript lo consume con `EventSource` o `fetch` con `ReadableStream`. Cada `data:` line es un chunk de texto que el cliente puede renderizar inmediatamente.

**El punto clave**: `.astream()` es la puerta de entrada al streaming en APIs. Sin async, tendrías que hacer `.stream()` en un thread separado, lo que elimina los beneficios de concurrencia.

---

## Por qué `build_query_chain()` no streamea

El módulo `rag/chain.py` tiene un `SemanticCache` que intercepta la respuesta:

```python
def query_fn(question: str):
    cached_answer = semantic_cache.get(question)
    if cached_answer is not None:
        return cached_answer, docs, True   # ← retorna string completo, no generador
    # ...
    answer = rag_chain.invoke(...)         # ← .invoke(), no .stream()
    return answer, docs, False
```

El cache funciona con respuestas completas. Para cachear haría falta acumular todos los chunks, guardarlo en cache, y LUEGO decidir si retornar el stream original o el valor cacheado. Es posible pero complica el diseño. La decisión de este proyecto es: el módulo de cache usa `.invoke()`, el script de streaming construye su propio pipeline minimal.

---

## Tradeoffs del streaming

### Ventajas

| Ventaja | Detalle |
|---------|---------|
| UX percibida mejor | El usuario ve texto desde el primer token (~300ms vs 5-15s) |
| Respuestas largas manejables | Sin streaming, el usuario no sabe si el sistema está colgado |
| Cancelación posible | Puedes cortar el stream antes de que termine si el usuario interrumpe |
| Integración natural con SSE | El protocolo HTTP tiene soporte nativo para streams de texto |

### Contra

| Contra | Detalle |
|--------|---------|
| Manejo de errores más complejo | Si el LLM falla a mitad del stream, ya enviaste parte del texto al cliente |
| Cache incompatible sin adaptación | No puedes cachear fácilmente una respuesta parcial |
| Métricas de latencia diferentes | La métrica útil es "tiempo al primer token" (TTFT), no "tiempo total" |
| Markdown roto en pantalla | Si renderizas Markdown en tiempo real, los bloques de código aparecen mal hasta que terminan |

### El problema del error a mitad del stream

```python
# Problema: si el LLM lanza una excepción en el chunk 50,
# el cliente ya recibió los primeros 49 chunks como texto válido.
# No puedes "deshacer" lo que ya enviaste.

# Estrategia defensiva: acumular + fallback
accumulated = ""
try:
    for chunk in chain.stream(input):
        accumulated += chunk
        yield chunk                     # enviar al cliente
except Exception as e:
    yield f"\n\n[Error: {e}]"          # al menos notificar al usuario
    # También: loguear accumulated para saber dónde falló
```

---

## Implementación en este proyecto

Ver [07_streaming.py](../../07_streaming.py) para la demo completa.

El pipeline de streaming construido tiene esta forma:

```
{"question": ..., "docs": [...]}
    │
    ▼
RunnablePassthrough.assign(context=format_docs)   ← bloquea, no streamea
    │
    ▼
QUERY_PROMPT (ChatPromptTemplate)                 ← bloquea, no streamea
    │
    ▼
ChatAnthropic                                     ← STREAMEA aquí
    │  token  │  token  │  token  │  ...
    ▼
StrOutputParser                                   ← pass-through, preserva el stream
    │
    ▼
generador de strings: "La" → " respuesta" → " es" → "..."
```
