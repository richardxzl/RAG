# Shared State vs Isolated State

## Los dos modelos

### Shared State — estado compartido

Todos los agentes leen y escriben en el mismo TypedDict.

```python
class EstadoCompartido(TypedDict):
    pregunta: str
    resultado_rag: str       # escrito por agente_rag
    resultado_analisis: str  # escrito por agente_analisis
    respuesta_final: str
    logs: Annotated[list[str], operator.add]
```

Cada agente tiene su propia key para evitar colisiones. Los nodos se ejecutan secuencialmente y cada uno puede leer lo que escribió el anterior.

### Isolated State — estado aislado (fan-out / fan-in)

Cada instancia de agente tiene su propio estado privado, creado con `Send()`.

```python
class EstadoPadre(TypedDict):
    pregunta: str
    perspectivas: list[str]
    resultados: Annotated[list[str], operator.add]  # reducer de acumulación

class EstadoAgente(TypedDict):
    pregunta: str
    perspectiva: str   # privado de esta instancia
    resultado: str
```

---

## Send() — fan-out paralelo

```python
from langgraph.types import Send

def nodo_fan_out(estado: EstadoPadre) -> list[Send]:
    # Crear N instancias del nodo "agente" en paralelo
    return [
        Send("agente", {"pregunta": estado["pregunta"], "perspectiva": p, "resultado": ""})
        for p in estado["perspectivas"]
    ]

def nodo_agente(estado: EstadoAgente) -> dict:
    resultado = llm.invoke(f"Responde como {estado['perspectiva']}: {estado['pregunta']}")
    # Este dict se acumula en el estado PADRE gracias a operator.add
    return {"resultados": [f"[{estado['perspectiva']}]: {resultado}"]}

# El fan-out se registra con add_conditional_edges desde START
builder.add_conditional_edges(START, nodo_fan_out, ["agente"])
builder.add_edge("agente", "fan_in")
```

---

## Cuándo usar cada uno

| Criterio | Shared State | Isolated State |
|---------|-------------|----------------|
| Los agentes necesitan ver el trabajo de otros | ✓ | — |
| Trabajo totalmente independiente | — | ✓ |
| Ejecución paralela | — | ✓ (Send) |
| Pipeline secuencial | ✓ | — |
| Resultado es agregación de N respuestas | — | ✓ |
| Estado simple | ✓ | — |

---

## Fan-in — agregar resultados

```python
def nodo_fan_in(estado: EstadoPadre) -> dict:
    # estado["resultados"] contiene todos los resultados acumulados
    todos = "\n\n".join(estado["resultados"])
    respuesta = llm.invoke(f"Sintetiza: {todos}").content
    return {"respuesta_final": respuesta}
```

El reducer `Annotated[list[str], operator.add]` garantiza que cada `Send` acumule su resultado sin sobrescribir los de los otros agentes.

---

## Patrón Map-Reduce

Fan-out + Fan-in es el patrón Map-Reduce aplicado a LLMs:
- **Map**: `Send()` despacha el mismo trabajo a N agentes (paralelizado)
- **Reduce**: el nodo fan-in agrega los N resultados en uno solo
