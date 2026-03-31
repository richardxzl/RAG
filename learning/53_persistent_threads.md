# Persistent Threads — Retomar conversaciones

## ¿Qué es un thread?

Un thread es una conversación identificada por un `thread_id`. Con un checkpointer persistente (SQLite, Postgres), el estado sobrevive reinicios del proceso.

```
Sesión 1:  grafo.invoke(Q1, thread="user-42")  → A1  [guardado en disco]
[proceso se reinicia]
Sesión 2:  grafo.invoke(Q2, thread="user-42")  → A2  [recuerda Q1 y A1]
```

---

## Flujo

```python
# 1. Checkpointer persistente
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
conn = sqlite3.connect("conversaciones.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# 2. Compilar con el checkpointer
grafo = builder.compile(checkpointer=checkpointer)

# 3. Cada invoke usa el mismo thread_id
cfg = {"configurable": {"thread_id": "user-42"}}
grafo.invoke({"messages": [HumanMessage(content="Hola")]}, config=cfg)

# [tiempo después, nuevo proceso]
grafo.invoke({"messages": [HumanMessage(content="¿Qué hablamos antes?")]}, config=cfg)
# → El LLM recibe el historial completo de la sesión anterior
```

---

## Obtener el estado de un thread

```python
cfg = {"configurable": {"thread_id": "user-42"}}
estado = grafo.get_state(cfg)

estado.values          # estado actual (mensajes, campos del TypedDict, etc.)
estado.next            # tupla de nodos pendientes (vacía si el grafo terminó)
estado.config          # config + checkpoint_id actual
estado.created_at      # timestamp del último checkpoint
```

---

## Múltiples threads simultáneos

```python
# Cada usuario tiene su propio thread — completamente aislados
for user_id in ["alice", "bob", "carlos"]:
    cfg = {"configurable": {"thread_id": f"user-{user_id}"}}
    grafo.invoke({"messages": [HumanMessage(content="Hola")]}, config=cfg)
```

Los threads no comparten estado entre sí, aunque usen el mismo grafo y el mismo checkpointer.

---

## Patrones de thread_id

```python
# Por usuario
thread_id = f"user-{user_id}"

# Por sesión (nuevo thread cada conversación)
import uuid
thread_id = str(uuid.uuid4())

# Por usuario + fecha (un thread por día)
from datetime import date
thread_id = f"user-{user_id}-{date.today()}"

# Por documento (conversación sobre un documento específico)
thread_id = f"user-{user_id}-doc-{doc_id}"
```

---

## MemorySaver vs checkpointers persistentes

`MemorySaver` guarda en RAM. Es perfecto para desarrollo pero **se pierde al reiniciar**.

Para threads que sobreviven reinicios:

```
pip install langgraph-checkpoint-sqlite   # SQLite
pip install langgraph-checkpoint-postgres # PostgreSQL
```
