"""
53_persistent_threads.py — Módulo 10.3: Persistent threads

Un "thread" en LangGraph es una conversación identificada por thread_id.
El checkpointer guarda el estado completo después de cada nodo,
permitiendo retomar la conversación exactamente donde quedó.

Casos de uso:
  - Chat multi-sesión: el usuario cierra la app y vuelve días después
  - Procesamiento pausado: job largo que se puede reanudar si falla
  - Auditoría: replay de conversaciones para debugging

Este script simula sesiones separadas (como si el proceso se reiniciara)
para demostrar que el estado persiste y se puede retomar.
"""
import logging
from typing import Annotated
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from rag.chain import get_llm

console = Console()

SYSTEM_PROMPT = (
    "Eres un asistente experto en IA. Responde en español, de forma concisa. "
    "Recuerda siempre el contexto previo de la conversación."
)


# ── Grafo de chat ─────────────────────────────────────────────────────────────

def nodo_chat(estado: MessagesState) -> dict:
    llm = get_llm()
    respuesta = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + estado["messages"])
    return {"messages": [respuesta]}


def construir_grafo(checkpointer):
    builder = StateGraph(MessagesState)
    builder.add_node("chat", nodo_chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


# ── Simular sesiones separadas ────────────────────────────────────────────────

def sesion(grafo, thread_id: str, mensajes: list[str], label: str):
    """Simula una sesión de conversación."""
    console.rule(f"[yellow]{label} (thread: '{thread_id}')")
    cfg = {"configurable": {"thread_id": thread_id}}

    for msg in mensajes:
        resultado = grafo.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config=cfg,
        )
        respuesta = resultado["messages"][-1].content
        n = len(resultado["messages"])
        console.print(f"  [dim][{n} msgs acumulados][/]")
        console.print(f"  [bold]Q:[/] {msg}")
        console.print(f"  [green]A:[/] {respuesta[:120]}...\n")


def inspeccionar_thread(grafo, thread_id: str):
    """Muestra el historial completo de un thread."""
    cfg = {"configurable": {"thread_id": thread_id}}
    estado = grafo.get_state(cfg)

    if not estado.values:
        console.print(f"  [dim]Thread '{thread_id}' vacío o no existe.[/]")
        return

    mensajes = estado.values.get("messages", [])
    tabla = Table(
        title=f"Historial del thread '{thread_id}'",
        show_header=True,
        header_style="bold magenta",
    )
    tabla.add_column("#")
    tabla.add_column("Rol")
    tabla.add_column("Contenido (100 chars)")

    for i, msg in enumerate(mensajes):
        rol = type(msg).__name__.replace("Message", "")
        tabla.add_row(str(i + 1), rol, msg.content[:100].replace("\n", " "))

    console.print(tabla)
    console.print(f"  [dim]Total: {len(mensajes)} mensajes en el historial[/]")


def listar_threads(checkpointer):
    """Lista todos los threads guardados en el checkpointer."""
    # MemorySaver guarda en memoria — no hay API de listado directo
    # En SQLite/Postgres sí hay API de listado, aquí lo simulamos
    console.print(Panel(
        "[bold]En producción (SQLite/Postgres), puedes listar todos los threads:[/]\n\n"
        "[dim]# Con SqliteSaver\n"
        "checkpoints = list(checkpointer.list(config=None))\n"
        "thread_ids = {c.config['configurable']['thread_id'] for c in checkpoints}\n\n"
        "# Con PostgresSaver\n"
        "async for checkpoint in checkpointer.alist(config):\n"
        "    print(checkpoint.config['configurable']['thread_id'])[/]",
        border_style="cyan",
    ))


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 10.3: Persistent Threads")

    console.print(Panel(
        "[bold]¿Qué es un thread?[/]\n\n"
        "  Un thread es una conversación identificada por [cyan]thread_id[/].\n"
        "  El checkpointer guarda el estado completo después de cada nodo.\n\n"
        "  Mismo grafo + mismo thread_id = misma conversación (continúa).\n"
        "  Mismo grafo + distinto thread_id = conversación nueva (aislada).\n\n"
        "[bold]Simularemos:[/]\n"
        "  Sesión 1 → usuario hace preguntas iniciales\n"
        "  [dim](el proceso 'se reinicia' — mismo checkpointer en memoria)[/]\n"
        "  Sesión 2 → el usuario retoma la conversación",
        border_style="blue",
        title="Persistent Threads",
    ))

    # Mismo checkpointer persiste entre 'sesiones'
    # En producción sería SqliteSaver o PostgresSaver
    checkpointer = MemorySaver()
    grafo = construir_grafo(checkpointer)

    THREAD = "usuario-ricardo-001"

    # Sesión 1: preguntas iniciales
    sesion(grafo, THREAD, [
        "¿Cuál es la diferencia entre un grafo y una chain en LangGraph?",
        "¿Puedes dar un ejemplo concreto de cuándo usar un grafo?",
    ], "Sesión 1 — primera visita")

    console.print(Panel(
        "[bold yellow]--- proceso 'reiniciado' ---[/]\n\n"
        "En producción: SqliteSaver o PostgresSaver persisten en disco.\n"
        "El mismo thread_id permite retomar la conversación.",
        border_style="yellow",
    ))

    # Sesión 2: retomar la conversación (referencias a lo anterior)
    sesion(grafo, THREAD, [
        "¿Y el ejemplo que mencionaste antes, cómo se implementaría con StateGraph?",  # requiere historia
        "Perfecto. ¿Qué módulo del roadmap cubre eso?",
    ], "Sesión 2 — retomando la conversación")

    # Inspeccionar el estado final
    inspeccionar_thread(grafo, THREAD)

    # Thread paralelo — completamente aislado
    sesion(grafo, "otro-usuario-999", [
        "¿Qué es LCEL?",
    ], "Thread distinto — no comparte historia")

    inspeccionar_thread(grafo, "otro-usuario-999")

    listar_threads(checkpointer)


if __name__ == "__main__":
    main()
