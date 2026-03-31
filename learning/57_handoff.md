# Handoff — Un agente pasa el control a otro

## ¿Qué es?

Handoff es cuando un agente activo transfiere el control a otro agente especializado. A diferencia del Supervisor (que orquesta desde arriba), en handoff los propios agentes deciden cuándo pasar el control.

```
Agente General → detecta pregunta técnica → HANDOFF → Agente Técnico
Agente General → detecta necesidad de docs → HANDOFF → Agente RAG
```

---

## Implementación con Command(goto=...)

```python
from langgraph.types import Command

def nodo_general(estado) -> Command:
    respuesta = llm.invoke(...)

    if "HANDOFF:agente_tecnico" in respuesta:
        # El nodo retorna un Command en vez de un dict
        return Command(
            goto="agente_tecnico",              # próximo nodo
            update={"agente_actual": "tecnico"} # cambios al estado
        )

    # Respuesta directa → ir a END
    return Command(
        goto=END,
        update={"messages": [AIMessage(content=respuesta)]}
    )
```

El nodo **retorna un `Command`** en vez de un dict. LangGraph salta al nodo indicado en `goto`.

---

## Grafo sin conditional_edges explícitos

Con Command, el routing está en el propio nodo — no necesitas `add_conditional_edges`:

```python
builder.add_node("agente_general", nodo_general)
builder.add_node("agente_tecnico", nodo_tecnico)
builder.add_node("agente_rag", nodo_rag)
builder.add_edge(START, "agente_general")
# No hay conditional_edges — el routing está en los Command
```

---

## Supervisor vs Handoff

| | Supervisor | Handoff |
|-|-----------|---------|
| Quién decide el routing | Nodo supervisor central | El propio agente activo |
| Modelo | Orquestación top-down | Peer-to-peer |
| Complejidad | Mayor (nodo extra) | Menor (el routing está en el nodo) |
| Flexibilidad | Alta (supervisor puede ver todo) | Alta (el agente decide en contexto) |
| Cuándo usar | Muchos agentes especializados | Pocos agentes con lógica clara |

---

## Command(goto) vs Command(goto=END)

```python
# Pasar a otro agente
return Command(goto="agente_tecnico", update={...})

# Terminar
return Command(goto=END, update={"respuesta_final": respuesta})

# Ir a un nodo con thread específico (multi-thread)
return Command(goto="agente_tecnico", update={...}, graph=Command.PARENT)
```
