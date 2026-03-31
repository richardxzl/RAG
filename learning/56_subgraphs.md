# Subgraphs — Grafos dentro de grafos

## ¿Qué es un subgraph?

Un grafo compilado que se usa como **nodo** dentro de otro grafo.

```python
subgraph = builder_interno.compile()   # compila el subgraph
grafo_padre.add_node("rag", subgraph)  # lo usa como nodo
```

El grafo padre no sabe ni le importa lo que hay dentro del subgraph — solo ve un nodo llamado "rag" que recibe y produce estado.

---

## Mapeo de estado

Para que el subgraph reciba y devuelva datos al grafo padre, las keys con **el mismo nombre** se mapean automáticamente:

```python
# Estado padre
class EstadoPadre(TypedDict):
    pregunta: str     # ← mismo nombre
    respuesta: str    # ← mismo nombre
    otro_campo: str

# Estado del subgraph
class EstadoSubgraph(TypedDict):
    pregunta: str     # ← mapeo automático desde padre
    documentos: list  # ← interno al subgraph
    respuesta: str    # ← mapeo automático hacia padre
```

Cuando el subgraph termina, `pregunta` y `respuesta` se copian de vuelta al estado padre. `documentos` queda encapsulado.

---

## Construir y usar

```python
# 1. Construir el subgraph
def construir_subgraph():
    builder = StateGraph(EstadoSubgraph)
    builder.add_node("retriever", fn_retriever)
    builder.add_node("generator", fn_generator)
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)
    return builder.compile()  # sin checkpointer propio

# 2. Usarlo en el grafo padre
subgraph = construir_subgraph()

builder_padre = StateGraph(EstadoPadre)
builder_padre.add_node("rag", subgraph)       # subgraph como nodo
builder_padre.add_node("otro", otro_nodo)
builder_padre.add_edge(START, "rag")
builder_padre.add_edge("rag", "otro")
```

---

## Visualizar subgraphs

```python
# Vista de superficie (subgraph como caja negra)
grafo_padre.get_graph().draw_ascii()

# Vista expandida (subgraph con sus nodos internos visibles)
grafo_padre.get_graph(xray=True).draw_ascii()
```

---

## Ventajas

- **Modularidad**: cada subgraph es una unidad testeable independientemente
- **Reutilización**: el mismo subgraph RAG puede usarse en varios grafos padre
- **Encapsulación**: el estado interno no contamina al padre
- **Legibilidad**: el grafo padre es simple — los detalles están adentro

---

## Cuándo usar subgraph vs nodo simple

| Usar nodo simple | Usar subgraph |
|-----------------|--------------|
| Lógica simple (1-2 pasos) | Lógica compleja (varios nodos) |
| Sin estado interno | Necesita estado interno propio |
| No se reutiliza | Se reutiliza en múltiples grafos |
| Prototipado rápido | Sistema modular en producción |
