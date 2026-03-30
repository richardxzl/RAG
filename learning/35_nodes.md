# Nodes — Funciones que transforman el estado

## Anatomía de un nodo

```python
def mi_nodo(estado: MiEstado) -> dict:
    # 1. Lee del estado
    valor = estado["alguna_key"]

    # 2. Hace algo (transforma, llama al LLM, busca en BD, etc.)
    resultado = transformar(valor)

    # 3. Retorna SOLO las keys que cambian
    return {"otra_key": resultado}
```

Los nodos son funciones Python normales. No heredan ninguna clase. Cualquier función con la firma correcta es un nodo.

---

## Merge parcial del estado

LangGraph **no reemplaza** el estado completo — hace un merge:

```python
# Estado actual
estado = {"a": 1, "b": 2, "c": 3}

# Nodo retorna
return {"b": 99}

# Estado resultante
estado = {"a": 1, "b": 99, "c": 3}  # solo 'b' cambió
```

Esto significa que los nodos son **compositivos**: no necesitan saber qué otras keys existen.

---

## Reducer pattern — Annotated

Por defecto, si un nodo retorna `{"logs": ["nuevo"]}`, la lista anterior se **reemplaza**. Para **acumular**, usa `Annotated`:

```python
from typing import Annotated
import operator

class Estado(TypedDict):
    # Sin Annotated: cada nodo reemplaza la lista
    historial_malo: list[str]

    # Con Annotated + operator.add: cada nodo ACUMULA
    logs: Annotated[list[str], operator.add]
```

```python
# nodo_a retorna: {"logs": ["a ejecutado"]}
# nodo_b retorna: {"logs": ["b ejecutado"]}
# Estado final:   logs = ["a ejecutado", "b ejecutado"]  ✓
```

---

## Tipos de nodos por responsabilidad

### Nodo de transformación pura
```python
def normalizar(estado: Estado) -> dict:
    return {"pregunta": estado["pregunta"].strip().lower() + "?"}
```

### Nodo con LLM
```python
def generar(estado: Estado) -> dict:
    respuesta = llm.invoke([HumanMessage(content=estado["pregunta"])]).content
    return {"respuesta": respuesta, "intentos": estado["intentos"] + 1}
```

### Nodo de efecto secundario (logging, métricas)
```python
def auditar(estado: Estado) -> dict:
    # Solo registra, no modifica datos clave
    log_entry = f"pregunta={estado['pregunta'][:40]}"
    return {"logs": [log_entry]}  # acumula en logs
```

### Nodo condicional (prepara la decisión)
```python
def clasificar(estado: Estado) -> dict:
    # Escribe 'tipo' para que la función router lo lea
    tipo = "rag" if "documento" in estado["pregunta"] else "directo"
    return {"tipo": tipo}
```

---

## Nodos asíncronos

LangGraph soporta nodos `async`:

```python
async def nodo_async(estado: Estado) -> dict:
    respuesta = await llm.ainvoke([HumanMessage(content=estado["pregunta"])])
    return {"respuesta": respuesta.content}

# Ejecutar
resultado = await grafo.ainvoke(estado_inicial)
```

---

## Qué NO debe hacer un nodo

- **No leer de fuera del estado**: si un nodo necesita datos, deben estar en el estado.
- **No retornar el estado completo** si solo cambia una key: es verboso y error-prone.
- **No llamar a otro nodo directamente**: los edges son responsabilidad del grafo, no de los nodos.
