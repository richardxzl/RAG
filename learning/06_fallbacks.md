# Fallbacks — Retry y modelo alternativo cuando falla un paso

## El problema que resuelven

Los LLMs y sus dependencias fallan. No es una excepción: es la norma en producción.

```
Anthropic API  → rate limit (429), timeout, modelo en mantenimiento
Redis cache    → conexión caída
ChromaDB       → disco lleno, pod reiniciando
Red            → timeout intermitente
```

Sin resiliencia, un fallo en cualquier punto del pipeline propaga una excepción al usuario. Con resiliencia, el sistema se degrada con gracia: reintenta, cambia de componente, o responde con menor calidad antes de renunciar.

LCEL provee dos mecanismos nativos: `.with_retry()` y `.with_fallbacks()`.

---

## `.with_retry()` — mismo Runnable, diferente intento

Envuelve cualquier Runnable para que reintente automáticamente cuando lanza excepción.

```python
from langchain_core.runnables import RunnableLambda

paso_inestable = RunnableLambda(mi_funcion).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
    retry_if_exception_type=(Exception,),
)
```

### Parámetros clave

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `stop_after_attempt` | `int` | `3` | Número máximo de intentos totales (no reintentos: 3 = 1 original + 2 reintentos) |
| `wait_exponential_jitter` | `bool` | `True` | Espera exponencial con jitter entre reintentos (evita thundering herd) |
| `retry_if_exception_type` | `tuple[type, ...]` | `(Exception,)` | Solo reintenta si la excepción es de este tipo |

### Cómo funciona internamente

```
intento 1 → falla → espera ~1s
intento 2 → falla → espera ~2s
intento 3 → éxito → retorna resultado
```

Con `wait_exponential_jitter=True`, LangChain usa `tenacity` bajo el capó con backoff exponencial más un valor aleatorio (jitter). Esto evita que todos los clientes reintenten al mismo tiempo después de un corte.

### Cuándo usarlo

- Rate limits (429): la API acepta la misma llamada si esperas un momento
- Timeouts de red: el servidor responde, pero la conexión tardó demasiado
- Errores 503 transitorios: el servicio estaba recargado, no caído permanentemente

### Cuándo NO usarlo

- Errores 401 (credenciales inválidas): reintentar no tiene sentido
- Input malformado: el mismo input dará el mismo error
- Errores de lógica de negocio propios

```python
# Reintenta SOLO ante errores de red, no ante cualquier excepción
from anthropic import APIConnectionError, RateLimitError

paso_llm = RunnableLambda(llamar_llm).with_retry(
    stop_after_attempt=4,
    retry_if_exception_type=(APIConnectionError, RateLimitError),
)
```

---

## `.with_fallbacks()` — Runnable diferente cuando el primario falla

Cuando el primer Runnable falla, LangChain prueba los fallbacks en orden hasta que uno tiene éxito.

```python
llm_primario = ChatAnthropic(model="claude-opus-4-5")
llm_economico = ChatAnthropic(model="claude-haiku-4-5")

llm_resiliente = llm_primario.with_fallbacks(
    fallbacks=[llm_economico],
    exceptions_to_handle=(Exception,),
)
```

### Parámetros

| Parámetro | Tipo | Descripción |
|---|---|---|
| `fallbacks` | `list[Runnable]` | Runnables alternativos, evaluados en orden |
| `exceptions_to_handle` | `tuple[type, ...]` | Solo activa el fallback para estas excepciones. Si omites este parámetro, maneja `Exception` por defecto |

### Lista de fallbacks: evaluación en orden

```python
# Si llm_a falla → prueba llm_b → si llm_b falla → prueba llm_c
llm_resiliente = llm_a.with_fallbacks(
    fallbacks=[llm_b, llm_c],
)
```

Si todos fallan, la última excepción se propaga al llamador.

### `exceptions_to_handle` — cuándo usarlo

Por defecto, `.with_fallbacks()` captura cualquier `Exception`. Eso puede ocultar errores que no deberían silenciarse:

```python
# Sin exceptions_to_handle: fallback activo ante CUALQUIER error
# Incluyendo errores de configuración, bugs propios, etc.
llm_a.with_fallbacks(fallbacks=[llm_b])

# Con exceptions_to_handle: fallback SOLO ante errores de la API
from anthropic import APIError

llm_a.with_fallbacks(
    fallbacks=[llm_b],
    exceptions_to_handle=(APIError,),
)
```

Regla práctica: sé específico con `exceptions_to_handle` en producción. Un error de programación no debería silenciarse con un fallback.

---

## retry vs fallback — la diferencia conceptual

| | `.with_retry()` | `.with_fallbacks()` |
|---|---|---|
| ¿Qué cambia entre intentos? | Nada — mismo Runnable, mismo input | Todo — Runnable diferente |
| ¿Cuándo aplica? | Error transitorio (el componente volverá) | Error permanente (el componente no responderá) |
| Latencia | Aumenta por las esperas entre reintentos | Baja: el fallback responde inmediatamente |
| Calidad de respuesta | Igual (mismo componente) | Puede ser menor (componente más simple) |
| Caso de uso típico | Rate limit, timeout de red | Modelo caído, servicio no disponible |

La combinación más potente es retry DENTRO del primario y fallback como red de seguridad:

```python
llm_primario = ChatAnthropic(model="claude-opus-4-5")

# Primero reintenta 3 veces; si sigue fallando, activa el fallback
llm_primario_con_retry = RunnableLambda(
    lambda x: llm_primario.invoke(x)
).with_retry(stop_after_attempt=3)

llm_economico = ChatAnthropic(model="claude-haiku-4-5")

llm_final = llm_primario_con_retry.with_fallbacks(
    fallbacks=[llm_economico],
)
```

---

## Fallback a nivel de pipeline completo

`.with_fallbacks()` funciona en cualquier Runnable, incluyendo un pipeline entero compuesto con `|`:

```python
pipeline_rag = retriever | prompt_rag | llm | StrOutputParser()
pipeline_llm = prompt_directo | llm | StrOutputParser()

pipeline_resiliente = pipeline_rag.with_fallbacks(
    fallbacks=[pipeline_llm],
)
```

Si cualquier paso del `pipeline_rag` falla, LangChain activa `pipeline_llm` completo con el input original.

Esto modela la degradación real:

```
ChromaDB disponible   → responde con contexto de documentos (máxima calidad)
ChromaDB caído        → responde solo con LLM (calidad reducida, pero funciona)
LLM también caído     → propaga la excepción (no hay más fallbacks)
```

---

## Casos de uso reales

### Rate limits

El escenario más común. Anthropic limita tokens por minuto. Con retry:

```python
from anthropic import RateLimitError

llm_con_retry = RunnableLambda(
    lambda x: llm.invoke(x)
).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True,
    retry_if_exception_type=(RateLimitError,),
)
```

### Modelo en mantenimiento → modelo de backup

Anthropic o cualquier proveedor puede sacar modelos de servicio. Un fallback a un modelo diferente mantiene el sistema operativo:

```python
llm_principal = ChatAnthropic(model="claude-opus-4-5")
llm_backup = ChatAnthropic(model="claude-haiku-4-5")
llm_minimo = ChatOpenAI(model="gpt-4o-mini")  # diferente proveedor

llm = llm_principal.with_fallbacks(
    fallbacks=[llm_backup, llm_minimo],
    exceptions_to_handle=(Exception,),
)
```

### Servicio de vectores caído → respuesta sin RAG

Cuando ChromaDB o el retriever no están disponibles, mejor responder sin contexto que no responder:

```python
rag_chain = retriever | prompt_rag | llm | parser
llm_chain = prompt_directo | llm | parser

chain = rag_chain.with_fallbacks(fallbacks=[llm_chain])
```

---

## Tradeoffs

### `.with_retry()`

| Ventaja | Contra |
|---|---|
| Transparente para el llamador | Aumenta la latencia (suma el tiempo de espera) |
| Maneja errores transitorios sin cambiar la lógica | No ayuda si el error es permanente |
| Configuración simple | Sin jitter, puede causar thundering herd |

### `.with_fallbacks()`

| Ventaja | Contra |
|---|---|
| Sistema siempre responde (degradación gracia) | El fallback puede dar respuesta de menor calidad |
| Composable: aplica a cualquier Runnable o pipeline | Más complejo de testear (múltiples rutas) |
| Fallbacks múltiples en cascada | Si todos fallan, la UX sigue siendo mala |

---

## Implementación en este proyecto

Ver [06_fallbacks.py](../../06_fallbacks.py) para la demo completa con tres casos:

```
Caso A: flaky_step (falla 2 veces) → .with_retry(3) → éxito en 3er intento
Caso B: llm_roto → .with_fallbacks([llm_real]) → respuesta del fallback
Caso C: retriever_roto → .with_fallbacks([pipeline_llm_directo]) → respuesta degradada
```

---

## Regla de oro

> `retry` cuando esperas que el componente se recupere.
> `fallback` cuando el componente no se recuperará pronto.
>
> Combínalos: reintenta primero para errores transitorios; si persiste, cambia de componente.
> Sé específico con `exceptions_to_handle` — no silencies errores de programación con fallbacks.
