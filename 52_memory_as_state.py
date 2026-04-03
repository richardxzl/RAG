"""
52_memory_as_state.py — Módulo 10.2: Memory as state

En LangGraph, la "memoria" de una conversación ES el estado del grafo.
La forma más idiomática es usar MessagesState: un TypedDict que contiene
una lista de BaseMessage con acumulación automática.

MessagesState vs estado personalizado:
  - MessagesState: lista de mensajes (HumanMessage, AIMessage, ToolMessage)
    → el LLM recibe toda la historia directamente
  - Estado personalizado con historial: más control, más código

Este script implementa un chat con memoria usando MessagesState y
compara con la aproximación de ventana de contexto (evitar tokens infinitos).
"""
import logging
from typing import Annotated

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from rag.chain import get_llm

console = Console()

SYSTEM_PROMPT = (
    "Eres un asistente experto en LangChain y LangGraph. "
    "Responde en español, de forma concisa y precisa. "
    "Recuerda el contexto de la conversación."
)


# ── Opción 1: MessagesState simple (sin límite de tokens) ─────────────────────

def nodo_chat_simple(estado: MessagesState) -> dict:
    """
    Pasa todos los mensajes al LLM. Simple, pero el contexto crece sin límite.
    En conversaciones largas, eventualmente excede el context window del LLM.
    """
    llm = get_llm()
    mensajes = [SystemMessage(content=SYSTEM_PROMPT)] + estado["messages"]
    respuesta = llm.invoke(mensajes)
    return {"messages": [respuesta]}


# ── Opción 2: Ventana de contexto (evitar tokens infinitos) ───────────────────

MAX_TOKENS = 2000  # límite conservador para Claude Haiku


def nodo_chat_con_ventana(estado: MessagesState) -> dict:
    """
    Recorta los mensajes antes de pasarlos al LLM.
    trim_messages mantiene los más recientes dentro del límite de tokens.
    Siempre mantiene el SystemMessage independientemente del recorte.
    """
    llm = get_llm()

    # Recortar mensajes manteniendo los más recientes
    mensajes_recortados = trim_messages(
        estado["messages"],
        max_tokens=MAX_TOKENS,
        token_counter=llm,             # usa el mismo LLM para contar tokens
        strategy="last",               # mantener los mensajes más recientes
        start_on="human",              # empezar en un mensaje humano
        include_system=True,           # no eliminar el SystemMessage
        allow_partial=False,
    )

    mensajes_con_system = [SystemMessage(content=SYSTEM_PROMPT)] + mensajes_recortados
    respuesta = llm.invoke(mensajes_con_system)
    return {"messages": [respuesta]}


# ── Construir grafos ──────────────────────────────────────────────────────────

def construir_grafo(con_ventana: bool = False):
    builder = StateGraph(MessagesState)
    nodo_fn = nodo_chat_con_ventana if con_ventana else nodo_chat_simple
    builder.add_node("chat", nodo_fn)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=MemorySaver())


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_conversacion(grafo, thread_id: str, preguntas: list[str]):
    cfg = {"configurable": {"thread_id": thread_id}}
    respuestas = []
    for pregunta in preguntas:
        resultado = grafo.invoke(
            {"messages": [HumanMessage(content=pregunta)]},
            config=cfg,
        )
        respuesta = resultado["messages"][-1].content
        respuestas.append(respuesta)
        n_mensajes = len(resultado["messages"])
        console.print(f"  [dim][{n_mensajes} msgs][/] [bold]Q:[/] {pregunta}")
        console.print(f"  [green]A:[/] {respuesta[:100]}...\n")
    return respuestas


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 10.2: Memory as State")

    console.print(Panel(
        "[bold]MessagesState:[/]\n\n"
        "  La memoria de la conversación ES el estado del grafo.\n"
        "  Cada mensaje (HumanMessage, AIMessage) se acumula en la lista.\n"
        "  El LLM recibe toda la historia → recuerda el contexto.\n\n"
        "[bold]Problema:[/] el contexto crece sin límite → context window overflow.\n"
        "[bold]Solución:[/] trim_messages — recortar a los mensajes más recientes.",
        border_style="blue",
        title="Memory as State",
    ))

    preguntas = [
        "¿Qué es LangGraph en una frase?",
        "¿Y cuál es la diferencia principal con LCEL que mencionaste antes?",   # requiere memoria
        "¿Cuándo usarías uno vs el otro, basándote en lo que dijiste?",        # requiere memoria
    ]

    # Demo 1: memoria simple
    console.rule("[yellow]Chat con memoria simple (MessagesState)")
    grafo_simple = construir_grafo(con_ventana=False)
    demo_conversacion(grafo_simple, "simple-1", preguntas)

    # Demo 2: memoria con ventana
    console.rule("[yellow]Chat con ventana de contexto (trim_messages)")
    grafo_ventana = construir_grafo(con_ventana=True)
    demo_conversacion(grafo_ventana, "ventana-1", preguntas)

    # Inspeccionar mensajes acumulados en el estado
    console.rule("[yellow]Estado interno del thread")
    cfg = {"configurable": {"thread_id": "simple-1"}}
    estado = grafo_simple.get_state(cfg)
    mensajes = estado.values["messages"]

    tabla = Table(title="Mensajes en el estado", show_header=True, header_style="bold magenta")
    tabla.add_column("#")
    tabla.add_column("Tipo")
    tabla.add_column("Contenido (80 chars)")
    for i, msg in enumerate(mensajes):
        tabla.add_row(
            str(i + 1),
            type(msg).__name__,
            msg.content[:80].replace("\n", " "),
        )
    console.print(tabla)

    console.print(Panel(
        "[bold]Estrategias de memoria a largo plazo:[/]\n\n"
        "  [cyan]trim_messages[/]     → ventana deslizante (últimos N tokens)\n"
        "  [cyan]summarize_messages[/]→ resumir historia antigua + mantener recientes\n"
        "  [cyan]InMemoryStore[/]     → memoria externa (facts, preferencias)\n\n"
        "[dim]El módulo 10.3 cubre persistent threads con thread_id estable.[/]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
