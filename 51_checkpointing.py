"""
51_checkpointing.py — Módulo 10.1: Checkpointing

El checkpointer es el mecanismo que persiste el estado del grafo entre ejecuciones.
Sin checkpointer: cada .invoke() empieza desde cero.
Con checkpointer:  cada .invoke() continúa desde donde quedó el thread.

El thread_id identifica la conversación. Dos threads distintos son
dos ejecuciones completamente independientes, aunque usen el mismo grafo.

Checkpointers disponibles:
  - MemorySaver:  RAM (se pierde al reiniciar) — ideal para desarrollo
  - SqliteSaver:  SQLite (persistente) — requiere langgraph-checkpoint-sqlite
  - PostgresSaver: Postgres — requiere langgraph-checkpoint-postgres
"""
import logging
from typing import TypedDict, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Estado ────────────────────────────────────────────────────────────────────

class EstadoChat(TypedDict):
    pregunta: str
    respuesta: str
    turno: int
    historial: Annotated[list[str], operator.add]


# ── Nodos ─────────────────────────────────────────────────────────────────────

def nodo_responder(estado: EstadoChat) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])
    contexto = format_docs(docs)
    llm = get_llm()
    respuesta = (QUERY_PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content

    entrada = f"[Turno {estado['turno'] + 1}] Q: {estado['pregunta'][:50]} → A: {respuesta[:60]}"
    return {
        "respuesta": respuesta,
        "turno": estado["turno"] + 1,
        "historial": [entrada],
    }


# ── Grafo con checkpointer ────────────────────────────────────────────────────

def construir_grafo():
    builder = StateGraph(EstadoChat)
    builder.add_node("responder", nodo_responder)
    builder.add_edge(START, "responder")
    builder.add_edge("responder", END)
    # El checkpointer se pasa en .compile(), no en add_node
    return builder.compile(checkpointer=MemorySaver())


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 10.1: Checkpointing")

    console.print(Panel(
        "[bold]Sin checkpointer:[/]  cada invoke() empieza desde cero\n"
        "[bold]Con checkpointer:[/]  el estado persiste entre invocaciones\n\n"
        "  grafo.invoke(input, config={'configurable': {'thread_id': 'abc'}})\n\n"
        "  → Primer invoke:  turno=0 → turno=1\n"
        "  → Segundo invoke: turno=1 → turno=2  (continúa donde quedó)\n"
        "  → Tercer invoke:  turno=2 → turno=3\n\n"
        "[dim]El thread_id identifica la conversación.\n"
        "Threads distintos son historiales completamente independientes.[/]",
        border_style="blue",
        title="Checkpointing",
    ))

    grafo = construir_grafo()

    # ── Thread 1: misma conversación, múltiples turnos ────────────────────────
    console.rule("[yellow]Thread 'usuario-1': múltiples turnos")
    cfg_1 = {"configurable": {"thread_id": "usuario-1"}}

    preguntas = [
        "¿De qué trata el documento?",
        "¿Cuáles son los temas principales?",
        "Dame un resumen en una frase",
    ]

    for pregunta in preguntas:
        estado_entrada = {
            "pregunta": pregunta,
            "respuesta": "",
            "turno": 0,       # el checkpointer sobreescribe con el valor real
            "historial": [],
        }
        resultado = grafo.invoke(estado_entrada, config=cfg_1)
        console.print(f"  [dim]Turno {resultado['turno']}[/] [bold]Q:[/] {pregunta[:50]}")
        console.print(f"  [green]A:[/] {resultado['respuesta'][:120]}...\n")

    # Ver el estado final del thread
    estado_guardado = grafo.get_state(cfg_1)
    console.print(Panel(
        f"[bold]Estado guardado del thread 'usuario-1':[/]\n\n"
        f"  turno:    {estado_guardado.values['turno']}\n"
        f"  historial ({len(estado_guardado.values['historial'])} entradas):\n" +
        "\n".join(f"    • {h}" for h in estado_guardado.values["historial"]),
        border_style="cyan",
        title="get_state(thread_id)",
    ))

    # ── Thread 2: completamente independiente ─────────────────────────────────
    console.rule("[yellow]Thread 'usuario-2': independiente del primero")
    cfg_2 = {"configurable": {"thread_id": "usuario-2"}}
    estado_entrada_2 = {"pregunta": "¿Qué información hay disponible?", "respuesta": "", "turno": 0, "historial": []}
    resultado_2 = grafo.invoke(estado_entrada_2, config=cfg_2)

    console.print(f"  [dim]Turno del thread-2:[/] {resultado_2['turno']} (empieza en 1, no en 4)")
    console.print(f"  [dim]Historial del thread-2:[/] {len(resultado_2['historial'])} entrada(s)")

    # Comparativa
    tabla = Table(title="Threads independientes", show_header=True, header_style="bold magenta")
    tabla.add_column("Thread")
    tabla.add_column("Turno final")
    tabla.add_column("Entradas en historial")
    tabla.add_row("usuario-1", str(grafo.get_state(cfg_1).values["turno"]), str(len(grafo.get_state(cfg_1).values["historial"])))
    tabla.add_row("usuario-2", str(grafo.get_state(cfg_2).values["turno"]), str(len(grafo.get_state(cfg_2).values["historial"])))
    console.print(tabla)

    # ── Checkpointers en producción ───────────────────────────────────────────
    console.print(Panel(
        "[bold]Para persistencia real (sobrevive reinicios):[/]\n\n"
        "[dim]# SQLite — requiere: pip install langgraph-checkpoint-sqlite\n"
        "from langgraph.checkpoint.sqlite import SqliteSaver\n"
        "import sqlite3\n"
        "conn = sqlite3.connect('estados.db', check_same_thread=False)\n"
        "checkpointer = SqliteSaver(conn)\n\n"
        "# PostgreSQL — requiere: pip install langgraph-checkpoint-postgres\n"
        "from langgraph.checkpoint.postgres import PostgresSaver\n"
        "checkpointer = PostgresSaver.from_conn_string('postgresql://...')\n\n"
        "grafo = builder.compile(checkpointer=checkpointer)[/]",
        border_style="magenta",
        title="Checkpointers persistentes",
    ))


if __name__ == "__main__":
    main()
