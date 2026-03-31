# Time Travel — Volver a un estado anterior

## ¿Qué es?

El checkpointer guarda un snapshot del estado después de **cada nodo**. Time travel = retroceder a cualquier snapshot y continuar desde ahí.

```
Turno 1: [snap-A] → Turno 2: [snap-B] → Turno 3: [snap-C] → Turno 4: [snap-D]
                                ↑
                    Puedes volver aquí y explorar una rama diferente
```

---

## Ver el historial de snapshots

```python
cfg = {"configurable": {"thread_id": "mi-thread"}}
snapshots = list(grafo.get_state_history(cfg))
# Lista en orden INVERSO (más reciente primero)

for snap in snapshots:
    print(snap.config["configurable"]["checkpoint_id"])
    print(snap.values)      # estado en ese punto
    print(snap.created_at)  # timestamp
```

---

## Volver a un snapshot específico

```python
# Obtener el checkpoint_id del snapshot al que quieres volver
snap_anterior = snapshots[2]   # snapshots[0] = más reciente
cp_id = snap_anterior.config["configurable"]["checkpoint_id"]

# Continuar desde ese punto
cfg_rollback = {"configurable": {
    "thread_id": "mi-thread",
    "checkpoint_id": cp_id,    # ← la clave
}}
grafo.invoke(nuevo_input, config=cfg_rollback)
```

---

## Casos de uso

### Rollback — deshacer el último turno

```python
# El snapshot penúltimo es el estado antes del último turno
snapshots = list(grafo.get_state_history(cfg))
snap_antes = snapshots[1]   # 0=actual, 1=antes del último nodo
cp_id = snap_antes.config["configurable"]["checkpoint_id"]

cfg_undo = {"configurable": {"thread_id": THREAD, "checkpoint_id": cp_id}}
grafo.invoke(nueva_pregunta, config=cfg_undo)
```

### Branching — explorar alternativas

```python
# Desde el mismo snapshot, invocar con inputs distintos
cfg_base = {"configurable": {"thread_id": THREAD, "checkpoint_id": cp_id}}

rama_a = grafo.invoke({"messages": [HumanMessage(content="Opción A")]}, config=cfg_base)
rama_b = grafo.invoke({"messages": [HumanMessage(content="Opción B")]}, config=cfg_base)
# Ambas ramas son independientes a partir de ese punto
```

### Replay — reproducir la ejecución

```python
# Iterar todos los snapshots en orden cronológico
snapshots = list(grafo.get_state_history(cfg))
for snap in reversed(snapshots):
    msgs = snap.values.get("messages", [])
    if msgs:
        print(f"[{type(msgs[-1]).__name__}] {msgs[-1].content[:60]}")
```

---

## Cuántos snapshots se guardan

Uno por **paso de ejecución del grafo** (entrada al grafo + después de cada nodo). Con un grafo de 3 nodos, cada `invoke()` genera ~4 snapshots.

Para limpiar snapshots viejos en producción:
```python
# SqliteSaver / PostgresSaver tienen TTL configurable
checkpointer = SqliteSaver(conn, ttl={"days": 30})
```
