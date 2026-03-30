# Edges — Condicionales y rutas entre nodos

## Tipos de edges

### 1. Edge directa

```python
builder.add_edge("nodo_a", "nodo_b")
```

Siempre va de A a B. Sin condiciones. El 95% de las conexiones son así.

---

### 2. Edge condicional

```python
builder.add_conditional_edges(
    "nodo_origen",    # después de este nodo...
    funcion_router,   # ...llama a esta función...
    {                 # ...y mapea su resultado a un nodo
        "opcion_1": "nodo_x",
        "opcion_2": "nodo_y",
    }
)
```

---

### 3. Edges especiales

```python
builder.add_edge(START, "primer_nodo")   # punto de entrada del grafo
builder.add_edge("ultimo_nodo", END)     # punto de salida del grafo
```

`START` y `END` son constantes importadas de `langgraph.graph`.

---

## La función de routing

La función que pasas a `add_conditional_edges` **no es un nodo** — es una función de decisión:

```python
def router(estado: Estado) -> str:
    # Lee el estado
    # Retorna un string que mapea al siguiente nodo
    return "rag" if estado["necesita_busqueda"] else "directo"
```

Características:
- **NO modifica el estado** — solo lee
- **Retorna un string** (key del mapa de destinos)
- Puede llamar al LLM para hacer una decisión compleja
- Puede usar `Literal` para tipado estricto

```python
from typing import Literal

def router(estado: Estado) -> Literal["rag", "directo", "error"]:
    ...
```

---

## Patrón típico: nodo + router

El patrón más común es separar la lógica de transformación (nodo) de la lógica de routing (función):

```python
# Nodo: clasifica y escribe el resultado en el estado
def clasificar(estado: Estado) -> dict:
    tipo = llm.invoke(f"Clasifica: {estado['pregunta']}").content
    return {"tipo": tipo}

# Función router: solo lee el estado
def router(estado: Estado) -> str:
    return estado["tipo"]  # lee lo que escribió el nodo

# Grafo
builder.add_node("clasificar", clasificar)
builder.add_conditional_edges("clasificar", router, {
    "rag": "nodo_rag",
    "directo": "nodo_directo",
})
```

---

## Múltiples destinos desde un nodo

Un nodo puede conectarse con `add_edge` a múltiples nodos si quieres ejecutarlos en **paralelo**:

```python
# nodo_a dispara nodo_b Y nodo_c en paralelo
builder.add_edge("nodo_a", "nodo_b")
builder.add_edge("nodo_a", "nodo_c")
```

LangGraph ejecuta nodos concurrentes cuando no hay dependencias entre ellos.

---

## Ciclos (loops)

Un edge puede apuntar hacia "atrás" en el grafo, creando un loop:

```python
def decidir_continuar(estado: Estado) -> str:
    if estado["intentos"] < 3 and not estado["respuesta_ok"]:
        return "reintentar"
    return "finalizar"

builder.add_conditional_edges("verificar", decidir_continuar, {
    "reintentar": "generar",   # vuelve al nodo anterior
    "finalizar": END,
})
```

Este es el patrón fundamental del **Corrective RAG**: si los chunks no son relevantes, reformula y vuelve a buscar.
