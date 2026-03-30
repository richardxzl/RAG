# RunnableLambda — Funciones custom como pasos del pipeline

## El problema que resuelve

LCEL conecta piezas con `|`. Cada pieza debe ser un **Runnable** — un objeto con `.invoke()`. Pero tu código no es un Runnable: es una función Python normal.

```python
# Esto NO funciona:
chain = retriever | format_docs | prompt | llm | StrOutputParser()
#                   ^^^^^^^^^^
#                   format_docs es una función, no un Runnable → TypeError
```

`RunnableLambda` es el adaptador. Convierte cualquier función Python en un Runnable compatible con LCEL:

```python
from langchain_core.runnables import RunnableLambda

format_docs_runnable = RunnableLambda(format_docs)

# Ahora sí:
chain = retriever | format_docs_runnable | prompt | llm | StrOutputParser()
```

---

## ¿Qué es exactamente?

`RunnableLambda` es un wrapper. Internamente hace esto:

```python
class RunnableLambda:
    def __init__(self, func):
        self.func = func

    def invoke(self, input, config=None):
        return self.func(input)

    def stream(self, input, config=None):
        yield self.invoke(input, config)

    def batch(self, inputs, config=None):
        return [self.invoke(i, config) for i in inputs]
```

La función que le pasas recibe el output del paso anterior y retorna el input para el siguiente.

---

## Casos de uso

### 1. Insertar lógica de transformación

```python
def normalize_question(question: str) -> str:
    return question.strip().lower().rstrip("?!.,")

chain = RunnableLambda(normalize_question) | retriever | prompt | llm
```

La pregunta se normaliza ANTES de llegar al retriever. Esto mejora el hit rate del cache semántico porque "¿De qué trata?" y "de que trata" quedan iguales.

### 2. Logging sin romper el pipeline

El patrón clave: **siempre retornar el valor recibido**.

```python
def log_step(name: str):
    def _log(value):
        print(f"[{name}] {type(value).__name__}: {str(value)[:60]}")
        return value  # ← CRÍTICO: no modificar, solo observar
    return _log

chain = (
    retriever
    | RunnableLambda(log_step("retriever"))   # log → pasa el valor
    | format_docs
    | RunnableLambda(log_step("context"))     # log → pasa el valor
    | prompt
    | llm
)
```

Puedes insertar y sacar estos logs sin cambiar la lógica del pipeline.

### 3. Enriquecer el output

```python
import time

def add_metadata(start_time: float):
    def _enrich(answer: str) -> dict:
        return {
            "answer": answer,
            "elapsed_ms": round((time.time() - start_time) * 1000),
        }
    return _enrich

start = time.time()
chain = retriever | prompt | llm | StrOutputParser() | RunnableLambda(add_metadata(start))

result = chain.invoke("mi pregunta")
print(result["answer"])      # → "La respuesta..."
print(result["elapsed_ms"])  # → 1432
```

### 4. Integrar código externo

Validación, llamadas a APIs externas, acceso a bases de datos — cualquier cosa:

```python
def validate_input(question: str) -> str:
    if len(question) < 5:
        raise ValueError(f"Pregunta demasiado corta: '{question}'")
    return question

def call_external_api(docs: list) -> list:
    # enriquecer los docs con metadata de otra fuente
    for doc in docs:
        doc.metadata["relevance_score"] = external_api.score(doc.page_content)
    return docs

chain = (
    RunnableLambda(validate_input)
    | retriever
    | RunnableLambda(call_external_api)
    | format_docs
    | prompt
    | llm
)
```

---

## Diferencia con `lambda` inline

LCEL también acepta lambdas directas en algunos contextos (como dentro de `.assign()`):

```python
# Lambda inline — funciona dentro de .assign()
RunnablePassthrough.assign(
    context=lambda x: format_docs(retriever.invoke(x["question"]))
)

# RunnableLambda — funciona en cualquier punto del pipe
chain = RunnableLambda(lambda x: x.strip()) | retriever | ...
```

La diferencia es el **contexto de uso**:

| | Lambda inline | RunnableLambda |
|---|---|---|
| Dentro de `.assign()` | ✅ | ✅ |
| Como paso del pipe (`\|`) | ❌ | ✅ |
| Streaming automático | ❌ | ✅ |
| Batch automático | ❌ | ✅ |
| Async (`.ainvoke()`) | ❌ | ✅ (con `async def`) |

---

## Soporte async

Si tu función hace I/O (base de datos, API externa), usa `async def`:

```python
import asyncio

async def fetch_extra_context(question: str) -> str:
    result = await some_async_api.search(question)
    return result

chain = RunnableLambda(fetch_extra_context) | prompt | llm

# Funciona con ainvoke sin bloquear el event loop:
answer = await chain.ainvoke("mi pregunta")
```

Si pasas una función `sync` a `RunnableLambda`, LangChain la ejecuta en un thread pool automáticamente cuando llamas `.ainvoke()`. No necesitas envolver manualmente con `asyncio.to_thread`.

---

## Implementación en este proyecto

Ver [04_runnable_lambda.py](../../04_runnable_lambda.py) para la demo completa.

El pipeline construido tiene esta forma:

```
pregunta
    │
    ▼
RunnableLambda(normalize_question)    ← transforma el input
    │
    ▼
RunnablePassthrough.assign(
    context = retriever
              │
              ▼
              RunnableLambda(log_docs)     ← observa sin modificar
              │
              ▼
              format_docs
              │
              ▼
              RunnableLambda(log_context)  ← observa sin modificar
)
    │
    ▼
QUERY_PROMPT
    │
    ▼
LLM
    │
    ▼
StrOutputParser
    │
    ▼
RunnableLambda(add_timing)            ← enriquece el output
    │
    ▼
{"answer": "...", "elapsed_ms": 1432}
```

---

## Tradeoffs

| Ventaja | Detalle |
|---------|---------|
| Composabilidad total | Cualquier función entra al pipe con un wrapper |
| Streaming/batch/async gratis | RunnableLambda hereda toda la interfaz Runnable |
| Insertar/sacar pasos fácil | Agregar logging o transformación sin tocar la lógica |
| Debugging granular | Cada paso es observable individualmente |

| Contra | Detalle |
|--------|---------|
| Overhead de wrapper | Una capa extra de indirección por cada función |
| Tipado implícito | La función recibe `Any` — no hay validación de tipos automática |
| Errores difíciles de trazar | Si la función falla, el traceback incluye internals de LangChain |

---

## Regla de oro

> Si necesitas insertar código Python en un pipeline LCEL, usa `RunnableLambda`.
> Si el código es solo transformación de un dict, considera si `.assign()` con lambda es suficiente.

La diferencia clave: `.assign()` **agrega keys** a un dict existente. `RunnableLambda` **reemplaza** el output completo del paso anterior.

```python
# .assign() → agrega "context" sin perder "question"
RunnablePassthrough.assign(context=lambda x: fetch_context(x["question"]))
# Output: {"question": "...", "context": "..."}

# RunnableLambda → reemplaza el output
RunnableLambda(lambda x: fetch_context(x["question"]))
# Output: solo el string del contexto, "question" se pierde
```
