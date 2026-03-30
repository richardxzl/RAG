# Compilar y ejecutar un grafo mínimo

## El proceso completo

```
StateGraph(Estado)      ← 1. Builder vacío con tipo de estado
  .add_node()           ← 2. Registrar nodos
  .add_edge()           ← 3. Conectar nodos
  .compile()            ← 4. Validar y compilar → CompiledGraph
  .invoke()             ← 5. Ejecutar
```

---

## Compilación — qué valida

`.compile()` no es solo un paso burocrático — valida:
- Que todos los nodos referenciados en edges existen
- Que el grafo tiene al menos una entrada (`START` o `set_entry_point`)
- Que no hay nodos sin edges de salida (nodos "muertos")
- Que el estado TypedDict tiene las keys correctas

```python
grafo = builder.compile()  # lanza ValueError si hay inconsistencias
```

---

## Formas de ejecución

### .invoke() — resultado final

```python
resultado = grafo.invoke(estado_inicial)
# Retorna el estado COMPLETO después de todos los nodos
# Bloqueante: espera a que termine
```

### .stream() — progreso por nodo

```python
# mode='updates' → solo las keys que cambió cada nodo
for evento in grafo.stream(estado, stream_mode="updates"):
    for nodo, cambios in evento.items():
        print(f"{nodo} actualizó: {list(cambios.keys())}")

# mode='values' → el estado completo después de cada nodo
for estado_actual in grafo.stream(estado, stream_mode="values"):
    print(f"Salida actual: {estado_actual.get('salida', '')}")
```

### .batch() — múltiples inputs en paralelo

```python
entradas = [
    {"pregunta": "¿Qué es Python?", ...},
    {"pregunta": "¿Qué es LangGraph?", ...},
]
resultados = grafo.batch(entradas)
# Ejecuta ambos en paralelo, retorna lista de estados finales
```

### Async

```python
resultado = await grafo.ainvoke(estado_inicial)

async for evento in grafo.astream(estado_inicial):
    ...
```

---

## Estado inicial

El estado inicial que pasas a `.invoke()` debe satisfacer el TypedDict:

```python
estado_inicial = {
    "pregunta": "¿Qué es LangGraph?",
    "respuesta": "",      # inicializar con valor vacío
    "intentos": 0,
    "logs": [],
}
```

**Gotcha**: si una key no está en el estado inicial, el primer nodo que intente leerla obtendrá `None` o un `KeyError`. Siempre inicializa todas las keys.

---

## Inspeccionar el grafo compilado

```python
graph_repr = grafo.get_graph()

# Lista de nodos
print([n.id for n in graph_repr.nodes.values()])

# Lista de edges
for edge in graph_repr.edges:
    print(f"{edge.source} → {edge.target}")

# ASCII art
print(graph_repr.draw_ascii())

# Mermaid
print(graph_repr.draw_mermaid())
```
