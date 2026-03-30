# Callbacks de LangChain — Hooks en el pipeline

## Qué son los callbacks

Los callbacks son observadores del pipeline. LangChain los invoca automáticamente en cada evento sin que tengas que modificar el código del pipeline:

```
chain.invoke(input, config={"callbacks": [mi_callback]})
                                              ↑
                         LangChain llama estos métodos automáticamente:
                         on_chain_start → on_retriever_start → on_retriever_end
                         → on_llm_start → on_llm_new_token (×N) → on_llm_end
                         → on_chain_end
```

---

## Implementar un callback

Hereda de `BaseCallbackHandler` y sobrescribe los métodos que necesitas:

```python
from langchain_core.callbacks import BaseCallbackHandler

class MiCallback(BaseCallbackHandler):

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM iniciado con modelo: {serialized.get('kwargs', {}).get('model')}")

    def on_llm_end(self, response: LLMResult, **kwargs):
        usage = response.llm_output.get("usage", {})
        print(f"Tokens usados: {usage}")

    def on_retriever_end(self, documents, **kwargs):
        print(f"Recuperados {len(documents)} docs")
```

---

## Eventos disponibles

| Evento | Cuándo se dispara |
|--------|------------------|
| `on_chain_start` | Inicio de cualquier Runnable |
| `on_chain_end` | Fin exitoso de un Runnable |
| `on_chain_error` | Error en un Runnable |
| `on_llm_start` | Inicio de llamada al LLM |
| `on_llm_new_token` | Nuevo token en streaming |
| `on_llm_end` | Fin de la llamada al LLM |
| `on_llm_error` | Error en la llamada al LLM |
| `on_retriever_start` | Inicio del retriever |
| `on_retriever_end` | Fin del retriever (con los docs) |
| `on_tool_start` | Inicio de una tool (agentes) |
| `on_tool_end` | Fin de una tool |

---

## Formas de registrar callbacks

```python
# Opción A: por invocación (solo para esta llamada)
chain.invoke(input, config={"callbacks": [mi_callback]})

# Opción B: en el componente (todas las invocaciones)
llm = ChatAnthropic(..., callbacks=[mi_callback])

# Opción C: callback global (todas las cadenas en el proceso)
from langchain_core.callbacks import set_global_handler
set_global_handler("stdout")
```

---

## Casos de uso

| Callback | Para qué |
|----------|---------|
| `TimingCallback` | Medir latencia por componente — detectar cuellos de botella |
| `LoggingCallback` | Audit log — qué queries llegaron, cuándo |
| `MetricsCallback` | Conteo de tokens y costos acumulados |
| `AlertCallback` | Disparar alertas si el LLM tarda > N segundos |
| `TracingCallback` | Enviar spans a Jaeger/Zipkin para distributed tracing |

---

## Callbacks vs stream_events

| | Callbacks | stream_events() |
|---|---|---|
| Tipo | Push (el framework te llama) | Pull (tú iteras) |
| Async | `AsyncCallbackHandler` | Nativo async |
| Granularidad | Por componente | Por evento individual |
| Cuándo usar | Observabilidad transversal | Streaming e inspección |

Los callbacks son mejor para **observabilidad** (logging, métricas, tracing). `stream_events()` es mejor para **UI** (mostrar progreso token a token).
