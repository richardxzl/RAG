# ReAct Pattern — Reasoning + Acting

## El patrón

ReAct (Wei et al., 2022) alterna entre razonamiento y acciones:

```
Thought:      "Necesito la población de ambas ciudades para compararlas"
Action:       buscar_poblacion("Ciudad de México")
Observation:  "~22 millones"
Thought:      "Ahora necesito la de Madrid"
Action:       buscar_poblacion("Madrid")
Observation:  "~6.7 millones"
Thought:      "Puedo calcular la razón ahora"
Action:       calcular("22000000 / 6700000")
Observation:  "3.2836"
Answer:       "Ciudad de México tiene ~3.3x la población de Madrid"
```

El loop continúa hasta que el LLM tiene suficiente información para responder.

---

## Implementación en LangGraph

El grafo ReAct son exactamente dos nodos con un loop:

```python
class EstadoReAct(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

def nodo_llm(estado):
    llm_con_tools = llm.bind_tools(tools)
    respuesta = llm_con_tools.invoke(estado["messages"])
    return {"messages": [respuesta]}

def necesita_tools(estado) -> str:
    ultimo = estado["messages"][-1]
    if isinstance(ultimo, AIMessage) and ultimo.tool_calls:
        return "tools"
    return END

builder.add_node("llm", nodo_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", necesita_tools, {"tools": "tools", END: END})
builder.add_edge("tools", "llm")   # ← el loop
```

---

## create_react_agent

Hace exactamente lo mismo en una línea:

```python
from langgraph.prebuilt import create_react_agent

agente = create_react_agent(llm, tools)
```

Usa `create_react_agent` para prototipado. Construye el grafo manual cuando necesitas:
- Añadir nodos extra (validación, logging, HITL)
- Cambiar la lógica de routing
- Personalizar el nodo LLM (prompt del sistema, temperatura por pregunta)

---

## ToolNode

`ToolNode` ejecuta todas las tools pendientes en `AIMessage.tool_calls`:

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
# Recibe el estado, ejecuta las tools, retorna ToolMessages
```

Ventajas sobre ejecutar las tools manualmente:
- Ejecuta múltiples tools en paralelo si hay varios tool_calls
- Maneja errores y los convierte en ToolMessages con el error
- Aplica throttling automático si está configurado

---

## El estado son mensajes

A diferencia de los grafos del módulo 8 (que usaban TypedDict con campos específicos), el agente ReAct usa una lista de mensajes como estado:

```python
messages: Annotated[list[BaseMessage], lambda x, y: x + y]
```

La historia completa de la conversación (HumanMessage → AIMessage → ToolMessage → AIMessage → ...) ES el estado del agente. Cada nodo lee el historial completo y agrega al final.

---

## Límite de pasos

Para evitar loops infinitos:

```python
agente = create_react_agent(
    llm,
    tools,
    recursion_limit=10,  # máximo de pasos en el loop
)
```

O manualmente en el grafo con un contador en el estado.
