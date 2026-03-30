# Human-in-the-Loop — Pausar y pedir confirmación

## ¿Por qué HITL?

Algunos agentes tienen herramientas que pueden tener consecuencias irreversibles: enviar emails, borrar datos, hacer compras. HITL permite pausar antes de esas acciones y pedir aprobación humana.

---

## Los tres componentes

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
```

| Componente | Rol |
|-----------|-----|
| `interrupt(value)` | Pausa el grafo y devuelve `value` al caller |
| `MemorySaver` | Persiste el estado pausado para poder reanudarlo |
| `Command(resume=v)` | Reanuda el grafo con el valor `v` |

---

## Flujo completo

```python
# 1. Construir el grafo con checkpointer
checkpointer = MemorySaver()
grafo = builder.compile(checkpointer=checkpointer)

# 2. Primera invocación — se pausará en interrupt()
thread_config = {"configurable": {"thread_id": "conv-1"}}
grafo.invoke({"messages": [HumanMessage(content=pregunta)]}, config=thread_config)

# 3. El grafo está pausado. Ver el estado:
estado = grafo.get_state(thread_config)
# estado.tasks[0].interrupts[0].value → el dato pasado a interrupt()

# 4. Reanudar con la respuesta del usuario
grafo.invoke(Command(resume="s"), config=thread_config)
```

---

## interrupt() dentro de un nodo

```python
def nodo_con_confirmacion(estado):
    for tool_call in estado["messages"][-1].tool_calls:
        if tool_call["name"] in TOOLS_PELIGROSAS:
            # PAUSA — el grafo se congela aquí
            aprobacion = interrupt({
                "accion": tool_call["name"],
                "args": tool_call["args"],
                "mensaje": f"¿Ejecutar {tool_call['name']}?",
            })
            # Cuando se reanude, 'aprobacion' tendrá el valor de Command(resume=...)

            if aprobacion.lower() not in ("s", "si", "y"):
                # Cancelar la tool
                continue
        # Ejecutar la tool
        ...
```

---

## interrupt_before (más simple)

Para pausar antes de CUALQUIER tool call sin modificar el nodo:

```python
agente = create_react_agent(
    llm,
    tools,
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pausa antes del nodo "tools"
)
```

---

## thread_id — identificar conversaciones pausadas

El `thread_id` es el identificador de la conversación. Permite tener múltiples conversaciones pausadas simultáneamente:

```python
# Conversación 1 pausada
grafo.invoke(pregunta_1, config={"configurable": {"thread_id": "user-alice"}})

# Conversación 2 pausada (independiente)
grafo.invoke(pregunta_2, config={"configurable": {"thread_id": "user-bob"}})

# Reanudar solo la conversación de Alice
grafo.invoke(Command(resume="s"), config={"configurable": {"thread_id": "user-alice"}})
```

---

## MemorySaver vs checkpointers persistentes

`MemorySaver` guarda el estado en RAM — se pierde al reiniciar el proceso.

Para producción, usar un checkpointer persistente:

```python
# PostgreSQL
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

# SQLite
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("sqlite:///estados.db")
```

---

## Cuándo usar HITL

| Escenario | Estrategia |
|-----------|-----------|
| Aprobar cualquier herramienta | `interrupt_before=["tools"]` |
| Solo tools peligrosas | `interrupt()` condicional en el nodo |
| Revisar respuesta antes de enviar | Nodo de revisión con `interrupt()` |
| Pedir clarificación | `interrupt()` cuando la intención es ambigua |
