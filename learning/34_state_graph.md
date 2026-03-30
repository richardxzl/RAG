# StateGraph — Estado, Nodos y Edges

## ¿Qué es LangGraph?

LangGraph es un framework para construir aplicaciones LLM con **estado**, **ciclos** y **múltiples actores**. Está construido sobre LangChain pero resuelve un problema que LCEL no puede: los **flujos no lineales**.

```
LCEL:    A → B → C → D       (siempre lineal)
LangGraph: A → B → C → A     (ciclos posibles)
            ↘ D ↗
```

---

## Los tres conceptos fundamentales

### 1. State (Estado)

Un `TypedDict` que define QUÉ datos viven en el grafo. Es la "memoria compartida" — todos los nodos leen y escriben en él.

```python
from typing import TypedDict

class MiEstado(TypedDict):
    mensaje: str
    pasos: list[str]
    contador: int
```

**Regla**: el estado debe contener TODO lo que cualquier nodo pueda necesitar.

---

### 2. Nodes (Nodos)

Funciones Python que **reciben el estado** y **retornan un dict parcial** con las keys a actualizar.

```python
def mi_nodo(estado: MiEstado) -> dict:
    return {
        "mensaje": estado["mensaje"].upper(),  # actualiza solo 'mensaje'
        "contador": estado["contador"] + 1,    # y 'contador'
        # 'pasos' no se toca → conserva su valor actual
    }
```

LangGraph hace un **merge shallow**: las keys que retornas reemplazan las del estado; las que no retornas se conservan.

---

### 3. Edges (Aristas)

Conexiones entre nodos. Hay tres tipos:

```python
# Directa: siempre va de A a B
builder.add_edge("nodo_a", "nodo_b")

# Condicional: una función decide el destino
builder.add_conditional_edges("nodo_a", funcion_router, {
    "opcion_1": "nodo_b",
    "opcion_2": "nodo_c",
})

# Especiales
builder.add_edge(START, "primer_nodo")  # entrada del grafo
builder.add_edge("ultimo_nodo", END)    # salida del grafo
```

---

## Flujo de construcción

```python
from langgraph.graph import StateGraph, START, END

# 1. Crear el builder
builder = StateGraph(MiEstado)

# 2. Registrar nodos
builder.add_node("nodo_a", funcion_a)
builder.add_node("nodo_b", funcion_b)

# 3. Conectar
builder.add_edge(START, "nodo_a")
builder.add_edge("nodo_a", "nodo_b")
builder.add_edge("nodo_b", END)

# 4. Compilar (valida la estructura)
grafo = builder.compile()

# 5. Ejecutar
estado_final = grafo.invoke({"mensaje": "hola", "pasos": [], "contador": 0})
```

---

## LCEL vs LangGraph

| Aspecto | LCEL | LangGraph |
|---------|------|-----------|
| Flujo | Lineal (`\|`) | Grafo (nodos + edges) |
| Estado | Dict pasado manualmente | TypedDict centralizado |
| Ciclos | No | Sí |
| Condicionales | RunnableBranch | Conditional edges |
| Persistencia | Manual | Checkpointing integrado |
| Multi-agente | No | Sí (subgraphs) |

**Cuándo usar cada uno:**
- Flujo lineal sin loops → LCEL es más simple
- Flujo con decisiones, loops, o múltiples agentes → LangGraph
