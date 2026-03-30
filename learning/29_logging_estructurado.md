# Logging Estructurado — Observabilidad por paso del pipeline

## Por qué JSON en vez de texto libre

```python
# ❌ Log de texto — difícil de parsear
logging.info("Retriever completado en 145ms, recuperó 4 docs de manual.pdf")

# ✅ Log estructurado — parseable por cualquier herramienta
logging.info("Paso completado", extra={
    "step": "retriever",
    "duration_ms": 145,
    "num_docs": 4,
    "request_id": "a3b7c9d2",
})
```

Con logs JSON puedes hacer queries como:
- "Todos los requests donde el retriever tardó > 500ms"
- "Requests que fallaron en el paso LLM"
- "Promedio de docs recuperados por hora"

---

## request_id — correlación de logs

Cada request genera un UUID único que aparece en **todos** los logs de esa request:

```json
{"request_id": "a3b7c9d2", "step": "retriever", "duration_ms": 145}
{"request_id": "a3b7c9d2", "step": "llm", "duration_ms": 1842}
{"request_id": "a3b7c9d2", "step": "request", "answer_chars": 312}
```

Esto permite reconstruir el timeline completo de una request específica en producción.

---

## Formatter estructurado

```python
class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Campos extra de logging.info("msg", extra={...})
        for key in ["request_id", "step", "duration_ms"]:
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry)

handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
```

---

## StepTimer — medir duración con context manager

```python
with StepTimer(logger, step="retriever", request_id=request_id):
    docs = retriever.invoke(question)
# Al salir del with: loguea duration_ms automáticamente
# Si hay excepción: loguea el error con el stack trace
```

El context manager es limpio — la lógica de timing no contamina la lógica de negocio.

---

## Niveles de logging en el pipeline RAG

| Nivel | Cuándo usarlo |
|-------|--------------|
| `DEBUG` | Detalles internos: longitud del contexto, tokens estimados |
| `INFO` | Eventos normales: paso completado, cache hit, request iniciada |
| `WARNING` | Situaciones inusuales: contexto vacío, query muy corto |
| `ERROR` | Fallos recuperables: timeout, parse error |
| `CRITICAL` | Fallos no recuperables: API key inválida, base de datos caída |

---

## Integración con herramientas de observabilidad

En producción, los logs JSON se envían a un agregador:

```python
# Loki (Grafana)
from logging_loki import LokiHandler
handler = LokiHandler(url="http://loki:3100/loki/api/v1/push", tags={"app": "rag"})

# Datadog
from datadog_logger import DatadogHandler
handler = DatadogHandler(api_key="...", service="rag-api")

logger.addHandler(handler)
```

Desde el agregador puedes crear dashboards, alertas, y análisis de rendimiento.
