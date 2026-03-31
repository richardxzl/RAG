# Memory as State — La conversación como estado del grafo

## El concepto

En LangGraph, la memoria de la conversación **ES** el estado. No hay un objeto `Memory` separado — los mensajes se acumulan en el estado del grafo como una lista.

```python
from langgraph.graph import MessagesState

# MessagesState es equivalente a:
class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # add_messages = reducer que acumula (no reemplaza) los mensajes
```

---

## Usar MessagesState

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

def nodo_chat(estado: MessagesState) -> dict:
    llm = get_llm()
    # El LLM recibe TODOS los mensajes previos → tiene memoria
    respuesta = llm.invoke([SystemMessage(content=SYSTEM)] + estado["messages"])
    return {"messages": [respuesta]}

builder = StateGraph(MessagesState)
builder.add_node("chat", nodo_chat)
# ...
grafo = builder.compile(checkpointer=MemorySaver())

# Cada invoke agrega mensajes al estado acumulado
cfg = {"configurable": {"thread_id": "conv-1"}}
grafo.invoke({"messages": [HumanMessage(content="Hola")]}, config=cfg)
grafo.invoke({"messages": [HumanMessage(content="¿Qué dijiste antes?")]}, config=cfg)
# El LLM recibe: [Human:"Hola", AI:"...", Human:"¿Qué dijiste antes?"]
```

---

## El problema del contexto infinito

Los mensajes crecen sin límite. Después de 50 turnos, el prompt puede superar el context window del LLM.

```
Turno 1:  [H, A]         → 200 tokens
Turno 10: [H,A,H,A...×10] → 2000 tokens
Turno 50: [H,A...×50]   → context window overflow ❌
```

---

## Solución: trim_messages

```python
from langchain_core.messages import trim_messages

def nodo_chat_con_ventana(estado: MessagesState) -> dict:
    llm = get_llm()
    mensajes_recortados = trim_messages(
        estado["messages"],
        max_tokens=2000,        # límite de tokens
        token_counter=llm,      # usa el LLM para contar
        strategy="last",        # mantener los más recientes
        start_on="human",       # empezar en mensaje humano
        include_system=True,    # no eliminar SystemMessage
    )
    return {"messages": [llm.invoke(mensajes_recortados)]}
```

---

## Estrategias de memoria a largo plazo

| Estrategia | Cómo | Cuándo |
|-----------|------|--------|
| Ventana deslizante | `trim_messages(max_tokens=N)` | Conversaciones largas sin resumen |
| Resumen | LLM resume los mensajes viejos | Cuando el historial es importante pero largo |
| Memoria externa | `InMemoryStore` con hechos clave | Preferencias, datos del usuario |

### Resumen de mensajes viejos

```python
def resumir_si_largo(estado: MessagesState) -> dict:
    if len(estado["messages"]) < 10:
        return {}  # sin cambios
    # Resumir los primeros N mensajes
    resumen = llm.invoke([
        SystemMessage(content="Resume esta conversación en 3 oraciones:"),
        *estado["messages"][:-4],
    ]).content
    # Reemplazar con el resumen + últimos 4 mensajes
    nuevos = [HumanMessage(content=f"Resumen previo: {resumen}")] + estado["messages"][-4:]
    return {"messages": nuevos}
```
