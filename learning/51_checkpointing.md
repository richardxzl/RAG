# Checkpointing — Persistir el estado entre ejecuciones

## El problema sin checkpointer

```python
grafo.invoke({"turno": 0})  # → turno = 1
grafo.invoke({"turno": 0})  # → turno = 1  (empieza desde cero otra vez)
```

Cada `invoke()` es independiente. El grafo no recuerda nada.

## Con checkpointer

```python
checkpointer = MemorySaver()
grafo = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "mi-conversacion"}}
grafo.invoke({"turno": 0}, config=cfg)  # → turno = 1
grafo.invoke({"turno": 0}, config=cfg)  # → turno = 2  (continúa)
```

El `thread_id` identifica la conversación. El checkpointer guarda el estado después de **cada nodo**.

---

## El thread_id

```python
# Misma conversación (continúa el estado)
cfg_alice = {"configurable": {"thread_id": "alice"}}
grafo.invoke(pregunta_1, config=cfg_alice)
grafo.invoke(pregunta_2, config=cfg_alice)  # recuerda pregunta_1

# Conversación nueva (estado aislado)
cfg_bob = {"configurable": {"thread_id": "bob"}}
grafo.invoke(pregunta_1, config=cfg_bob)    # no sabe nada de alice
```

---

## Inspeccionar el estado guardado

```python
estado = grafo.get_state(cfg)
print(estado.values)           # estado actual del thread
print(estado.next)             # próximo nodo a ejecutar (si está pausado)
print(estado.config)           # configuración + checkpoint_id actual
```

---

## Checkpointers disponibles

| Backend | Persistencia | Cuándo usar |
|---------|-------------|-------------|
| `MemorySaver` | RAM (se pierde al reiniciar) | Desarrollo, tests |
| `SqliteSaver` | SQLite en disco | Proyectos pequeños, single-process |
| `PostgresSaver` | PostgreSQL | Producción, multi-proceso |

```python
# MemorySaver (siempre disponible)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# SqliteSaver — pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
conn = sqlite3.connect("estados.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# PostgresSaver — pip install langgraph-checkpoint-postgres
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://user:pass@host/db")
```

---

## Qué se guarda exactamente

El checkpointer guarda después de **cada nodo**:
- El estado completo (todos los campos del TypedDict)
- El `checkpoint_id` (identificador único del snapshot)
- El `thread_id`
- Metadatos (qué nodo acaba de ejecutarse, cuál es el siguiente)

Esto es lo que permite el **time travel** (módulo 10.4).
