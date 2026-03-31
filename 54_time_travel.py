"""
54_time_travel.py — Módulo 10.4: Time travel — volver a un estado anterior

El checkpointer guarda un snapshot del estado después de CADA nodo.
Time travel = retroceder a cualquier snapshot y continuar desde ahí.

Casos de uso:
  - Debugging: reproducir exactamente cómo llegó el grafo a un estado incorrecto
  - Branching: explorar diferentes respuestas desde el mismo punto
  - Rollback: deshacer el último turno de una conversación
  - Testing: probar el mismo estado con diferentes continuaciones

APIs clave:
  - grafo.get_state_history(config)  → todos los snapshots del thread
  - grafo.get_state(config)          → snapshot más reciente
  - grafo.invoke(input, config)      → continuar desde un checkpoint_id específico
"""
from typing import Annotated

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from rag.chain import get_llm

console = Console()

SYSTEM_PROMPT = "Eres un asistente experto en IA. Responde en español, de forma breve."


# ── Grafo ─────────────────────────────────────────────────────────────────────

def nodo_chat(estado: MessagesState) -> dict:
    llm = get_llm()
    respuesta = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + estado["messages"])
    return {"messages": [respuesta]}


def construir_grafo():
    builder = StateGraph(MessagesState)
    builder.add_node("chat", nodo_chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=MemorySaver())


# ── Helpers ───────────────────────────────────────────────────────────────────

def chat(grafo, thread_id: str, pregunta: str) -> str:
    cfg = {"configurable": {"thread_id": thread_id}}
    resultado = grafo.invoke(
        {"messages": [HumanMessage(content=pregunta)]},
        config=cfg,
    )
    return resultado["messages"][-1].content


def mostrar_historial(grafo, thread_id: str):
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshots = list(grafo.get_state_history(cfg))

    tabla = Table(
        title=f"Snapshots del thread '{thread_id}'",
        show_header=True,
        header_style="bold magenta",
    )
    tabla.add_column("#")
    tabla.add_column("Checkpoint ID (10 chars)")
    tabla.add_column("Mensajes")
    tabla.add_column("Último mensaje (60 chars)")

    for i, snap in enumerate(snapshots):
        msgs = snap.values.get("messages", [])
        ultimo = msgs[-1].content[:60].replace("\n", " ") if msgs else "—"
        tabla.add_row(
            str(i),
            snap.config["configurable"].get("checkpoint_id", "?")[:10],
            str(len(msgs)),
            ultimo,
        )
    console.print(tabla)
    return snapshots


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]RAG Lab — Módulo 10.4: Time Travel")

    console.print(Panel(
        "[bold]El checkpointer guarda un snapshot tras CADA nodo.[/]\n\n"
        "  get_state_history(config) → lista de snapshots (más reciente primero)\n"
        "  Cada snapshot tiene un [cyan]checkpoint_id[/] único\n\n"
        "[bold]Time travel = continuar desde cualquier snapshot:[/]\n\n"
        "[dim]  cfg = {'configurable': {\n"
        "      'thread_id': 'mi-thread',\n"
        "      'checkpoint_id': 'snap-anterior',   # ← la clave\n"
        "  }}\n"
        "  grafo.invoke(nuevo_input, config=cfg)[/]",
        border_style="blue",
        title="Time Travel",
    ))

    grafo = construir_grafo()
    THREAD = "time-travel-demo"

    # Construir una conversación de varios turnos
    console.rule("[yellow]Construyendo historial (4 turnos)")
    preguntas = [
        "¿Qué es un StateGraph?",
        "¿Cuántos tipos de edges hay?",
        "¿Qué es un checkpointer?",
        "¿Para qué sirve el time travel?",
    ]
    for p in preguntas:
        respuesta = chat(grafo, THREAD, p)
        console.print(f"  [bold]Q:[/] {p}  [green]A:[/] {respuesta[:80]}...")

    # Ver todos los snapshots
    console.rule("[yellow]Historial de snapshots")
    snapshots = mostrar_historial(grafo, THREAD)

    # ── Demo 1: rollback — deshacer el último turno ───────────────────────────
    console.rule("[yellow]Demo 1: Rollback — deshacer el último turno")

    # El snapshot en índice 2 es el estado ANTES del turno 4
    # (snapshots están en orden inverso: 0=más reciente)
    if len(snapshots) >= 3:
        snap_anterior = snapshots[2]   # 2 turnos atrás
        checkpoint_id_anterior = snap_anterior.config["configurable"]["checkpoint_id"]
        n_msgs_antes = len(snap_anterior.values.get("messages", []))

        console.print(f"  Volviendo al snapshot con {n_msgs_antes} mensajes...")
        console.print(f"  checkpoint_id: {checkpoint_id_anterior[:20]}...")

        cfg_rollback = {"configurable": {
            "thread_id": THREAD,
            "checkpoint_id": checkpoint_id_anterior,
        }}

        # Continuar desde ese punto con una pregunta diferente
        resultado_rollback = grafo.invoke(
            {"messages": [HumanMessage(content="¿Y qué es el time travel exactamente?")]},
            config=cfg_rollback,
        )
        console.print(Panel(
            f"[dim]Estado tiene {len(resultado_rollback['messages'])} mensajes "
            f"(debería ser ~{n_msgs_antes + 2})[/]\n\n"
            + resultado_rollback["messages"][-1].content[:300],
            title="Respuesta desde el snapshot anterior",
            border_style="green",
        ))

    # ── Demo 2: branching — dos respuestas desde el mismo punto ──────────────
    console.rule("[yellow]Demo 2: Branching — explorar desde el mismo punto")

    if len(snapshots) >= 4:
        snap_base = snapshots[3]    # primer turno completado
        cp_id = snap_base.config["configurable"]["checkpoint_id"]
        cfg_branch = {"configurable": {"thread_id": THREAD, "checkpoint_id": cp_id}}

        rama_a = grafo.invoke(
            {"messages": [HumanMessage(content="Explícame los edges directos")]},
            config=cfg_branch,
        )
        rama_b = grafo.invoke(
            {"messages": [HumanMessage(content="Explícame los edges condicionales")]},
            config=cfg_branch,
        )

        tabla_branch = Table(title="Dos ramas desde el mismo checkpoint", show_header=True)
        tabla_branch.add_column("Rama")
        tabla_branch.add_column("Pregunta")
        tabla_branch.add_column("Respuesta (100 chars)")
        tabla_branch.add_row("A", "edges directos", rama_a["messages"][-1].content[:100])
        tabla_branch.add_row("B", "edges condicionales", rama_b["messages"][-1].content[:100])
        console.print(tabla_branch)

    # ── Demo 3: replay completo ───────────────────────────────────────────────
    console.rule("[yellow]Demo 3: Replay — reproducir la conversación paso a paso")
    cfg_original = {"configurable": {"thread_id": THREAD}}
    todos_los_snapshots = list(grafo.get_state_history(cfg_original))

    console.print(f"  Total de snapshots en el thread: {len(todos_los_snapshots)}")
    console.print("  [dim]Reproduciendo en orden cronológico (inverso al historial):[/]")

    for snap in reversed(todos_los_snapshots):
        msgs = snap.values.get("messages", [])
        if msgs:
            ultimo = msgs[-1]
            rol = "Human" if isinstance(ultimo, HumanMessage) else "AI"
            console.print(f"    [{rol}] {ultimo.content[:70]}...")


if __name__ == "__main__":
    main()
