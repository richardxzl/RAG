# Tracing — Visibilidad completa del pipeline

## Logging vs Tracing

| | Logging | Tracing |
|---|---|---|
| **Estructura** | Líneas de texto/JSON | Árbol de spans jerarquico |
| **Relación** | Eventos independientes | Causa → efecto |
| **Visualización** | Grep / dashboard | Flame graphs, timelines |
| **Debug** | "¿Qué pasó?" | "¿Por qué tardó tanto?" y "¿Qué llamó a qué?" |

El tracing captura el **árbol de ejecución**: qué llamó a qué, con qué inputs/outputs, cuánto tardó cada nodo.

---

## LangSmith — Tracing nativo de LangChain

Con solo configurar estas variables de entorno, LangChain traza **automáticamente** todo:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_PROJECT=mi-proyecto
```

```python
# Sin cambiar nada en el código:
chain.invoke({"question": "¿qué es LCEL?"})
# → Aparece en https://smith.langchain.com con el árbol completo
```

LangSmith captura: inputs/outputs de cada nodo, tokens usados, latencia, errores, y permite hacer "replay" de una request para debugear.

---

## Estructura de un span

Un span representa una unidad de trabajo:

```python
@dataclass
class Span:
    trace_id: str      # ID de la request completa
    span_id: str       # ID de este paso específico
    name: str          # "retriever", "llm", "format_docs"
    start_time: float
    end_time: float
    inputs: dict       # qué recibió
    outputs: dict      # qué retornó
    error: Optional[str]
    children: list[Span]  # subpasos
    parent_id: Optional[str]
```

El `trace_id` es el mismo para todos los spans de una request. El `parent_id` construye el árbol.

---

## Árbol de trazas del pipeline RAG

```
rag_query (1847ms)
  ├── retriever (145ms)
  │     ↳ input: {"query": "¿qué es LCEL?"}
  │     ↳ output: {"num_docs": 4}
  │
  ├── format_docs (2ms)
  │     ↳ output: {"context_chars": 3240}
  │
  └── llm (1700ms)          ← el cuello de botella
        ↳ output: {"answer_chars": 312}
```

Con este árbol puedes identificar inmediatamente que el LLM consume el 92% del tiempo.

---

## Alternativas a LangSmith

| Herramienta | Tipo | Cuándo usar |
|-------------|------|-------------|
| LangSmith | SaaS (LangChain) | Mejor integración, UI excelente |
| Phoenix (Arize) | Open source | Self-hosted, OTEL nativo |
| Langfuse | Open source SaaS | Alternativa a LangSmith |
| OpenTelemetry | Estándar | Si ya tienes infra OTEL (Jaeger, Tempo) |
| Tracer local (este módulo) | Custom | Aprendizaje, proyectos pequeños |

---

## Tracing en producción

El tracing tiene un costo de overhead (serializar spans, enviarlos a la red). Para mitigarlo:

```python
# Sampling: solo trazar el 10% de los requests en producción
import random
callbacks = [tracer] if random.random() < 0.1 else []
chain.invoke(input, config={"callbacks": callbacks})
```

Para debugging, activa el 100% temporalmente y vuelve al sampling después.
