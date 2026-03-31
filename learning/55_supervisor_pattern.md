# Supervisor Pattern — Un agente coordina a otros

## ¿Qué es?

El Supervisor es un agente central que:
1. Recibe la tarea
2. Decide a qué sub-agente delegarla
3. Recibe el resultado
4. Decide si está completo o delega de nuevo

```
START → Supervisor → agente_rag    → Supervisor → END (si FINISH)
                  → agente_calculo → Supervisor
                  → agente_resumen → Supervisor
```

**Regla clave**: todos los sub-agentes devuelven el control al supervisor. Solo el supervisor puede ir a END.

---

## Implementación

```python
class EstadoSupervisor(TypedDict):
    tarea: str
    siguiente: str           # "agente_X" o "FINISH"
    resultado_parcial: str
    mensajes: Annotated[list[str], operator.add]

def nodo_supervisor(estado) -> dict:
    # El LLM decide quién trabaja a continuación
    decision = llm.invoke(SUPERVISOR_PROMPT.format(...)).content
    return {"siguiente": decision}

def routing_supervisor(estado) -> str:
    return END if estado["siguiente"] == "FINISH" else estado["siguiente"]

# Todos los sub-agentes vuelven al supervisor
builder.add_edge("agente_rag", "supervisor")
builder.add_edge("agente_calculo", "supervisor")
builder.add_conditional_edges("supervisor", routing_supervisor, {
    "agente_rag": "agente_rag",
    "agente_calculo": "agente_calculo",
    END: END,
})
```

---

## El prompt del supervisor

El supervisor necesita saber:
1. Qué agentes tiene disponibles (con sus responsabilidades)
2. Qué se ha hecho ya (historial)
3. El resultado del último agente
4. Cuándo debe terminar (`FINISH`)

```python
SUPERVISOR_PROMPT = """
Agentes disponibles:
- agente_rag: busca en documentos
- agente_analisis: analiza datos

Historial: {historial}
Resultado último agente: {resultado}

¿Qué hacer? Responde SOLO: agente_rag | agente_analisis | FINISH
"""
```

---

## Supervisor vs ReAct

| | Supervisor | ReAct |
|-|-----------|-------|
| Quién decide | Nodo supervisor (LLM) | El mismo LLM del agente |
| Sub-agentes | Nodos completos (grafos) | Tools simples |
| Paralelismo | Posible (varios agentes a la vez) | Secuencial |
| Cuándo usar | Sub-tareas complejas con agentes especializados | Tools simples, un solo agente |
