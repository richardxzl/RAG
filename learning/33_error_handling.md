# Error Handling — Graceful Degradation del pipeline RAG

## El principio

El usuario nunca debe ver un stack trace de Python. Cada componente del pipeline puede fallar — el sistema debe degradarse con gracia, manteniendo la máxima funcionalidad posible.

```
Nivel 3 (óptimo):   cache hit            → respuesta instantánea
Nivel 2 (normal):   RAG completo         → respuesta con contexto
Nivel 1 (degradado): solo LLM            → respuesta sin contexto del documento
Nivel 0 (error):    mensaje claro        → "inténtalo más tarde"
```

---

## Redis down — cache opcional

```python
def get_cache_seguro():
    try:
        cache = SemanticCache()
        cache.stats()  # test de conexión
        return cache
    except Exception:
        return None  # el sistema funciona sin cache

cache = get_cache_seguro()

if cache:
    cached = cache.get(question)
    if cached:
        return cached  # fast path

# Continuar con RAG aunque el cache esté caído
answer = rag_chain.invoke(question)

if cache:
    try:
        cache.set(question, answer)
    except Exception:
        pass  # falla silenciosa al escribir en cache
```

**Patrón**: el cache es siempre opcional. Si no está disponible, el pipeline no se rompe — solo es más lento.

---

## ChromaDB no disponible — RAG opcional

```python
def query_con_rag_opcional(question: str) -> str:
    # Intento 1: RAG completo
    try:
        docs = retriever.invoke(question)
        return rag_chain.invoke({"question": question, "docs": docs})
    except Exception as e:
        log.warning(f"RAG falló: {e}")

    # Intento 2: LLM sin contexto (calidad menor)
    try:
        return llm_directo.invoke(question)
    except Exception:
        pass

    # Nivel 0: error total
    return FALLBACK_ERROR_TOTAL
```

---

## Mapeo de errores técnicos a mensajes de usuario

Los usuarios no entienden `anthropic.AuthenticationError`. Debes traducir:

```python
def mensaje_usuario(error: Exception) -> str:
    err = str(error).lower()

    if "api_key" in err or "401" in err:
        return "Error de configuración. Contacta al administrador."

    if "rate_limit" in err or "429" in err:
        return "Servicio saturado. Espera unos segundos e inténtalo de nuevo."

    if "timeout" in err:
        return "La respuesta tardó demasiado. Prueba con una pregunta más corta."

    if "context_length" in err or "too long" in err:
        return "La pregunta es demasiado larga. Por favor, sé más específico."

    return "Error técnico temporal. Inténtalo de nuevo en unos minutos."
```

---

## Errores que NO debes silenciar

No todo error debe tener un fallback. Algunos errores indican problemas que hay que resolver:

| Error | Acción correcta |
|-------|----------------|
| `ANTHROPIC_API_KEY` no configurada | Lanzar excepción en startup, no silenciar |
| ChromaDB corrompida | Alertar, no servir respuestas inventadas |
| Disco lleno | Alertar inmediatamente |
| Error de código (ValueError, TypeError) | Propagar — es un bug, no una degradación |

**Regla**: silencia los errores de **infraestructura** (caídas temporales). Propaga los errores de **código** (bugs).

---

## Circuit breaker — evitar cascadas de fallos

Si Redis está caído y cada request intenta conectarse (y falla), el timeout de conexión se acumula:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failures = 0
        self.last_failure = None
        self.threshold = failure_threshold
        self.timeout = timeout
        self.open = False

    def call(self, fn, *args):
        if self.open:
            if time.time() - self.last_failure > self.timeout:
                self.open = False  # half-open: intentar de nuevo
            else:
                raise Exception("Circuit breaker abierto")
        try:
            result = fn(*args)
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.open = True  # no más intentos por `timeout` segundos
            raise
```

Después de N fallos consecutivos, el circuit breaker deja de intentar la conexión durante un tiempo — protege al sistema de timeouts en cascada.
