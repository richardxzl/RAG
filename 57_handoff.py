"""
57_handoff.py — Módulo 11.3: Handoff — un agente pasa el control a otro

Handoff es el mecanismo por el cual un agente activo transfiere
el control de la conversación a otro agente especializado.

En LangGraph se implementa con Command(goto="nodo_destino"):
  - El agente actual genera un Command en vez de un AIMessage
  - El grafo salta al nodo especificado en goto
  - El agente destino recibe el estado completo y continúa

Diferencia con el Supervisor:
  Supervisor: coordina desde arriba (orquestación top-down)
  Handoff:    el agente mismo decide pasar el control (peer-to-peer)

Este script implementa un sistema de soporte técnico donde:
  - agente_general: atiende la consulta inicial
  - agente_tecnico: experto en LangGraph/LangChain
  - agente_rag: accede a la base de conocimiento
"""
import logging
from typing import TypedDict, Annotated, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Estado ────────────────────────────────────────────────────────────────────

class EstadoHandoff(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    agente_actual: str
    historial_handoffs: Annotated[list[str], lambda x, y: x + y]


# ── Prompts de cada agente ────────────────────────────────────────────────────

PROMPT_GENERAL = ChatPromptTemplate.from_template(
    """Eres un agente de soporte general. Eres amigable pero reconoces tus límites.

Si la pregunta requiere acceder a documentos específicos → responde SOLO: HANDOFF:agente_rag
Si la pregunta es técnica sobre LangGraph/LangChain → responde SOLO: HANDOFF:agente_tecnico
Si puedes responder tú mismo → da la respuesta directamente.

Conversación:
{mensajes}

Pregunta actual: {pregunta}

Respuesta (o HANDOFF:destino):"""
)

PROMPT_TECNICO = ChatPromptTemplate.from_template(
    """Eres un experto técnico en LangChain y LangGraph. Das respuestas precisas y detalladas.
Si la pregunta requiere buscar en documentos → responde SOLO: HANDOFF:agente_rag
Si ya puedes responder → da la respuesta técnica completa.

Contexto: {contexto}
Pregunta: {pregunta}

Respuesta técnica:"""
)

PROMPT_RAG = ChatPromptTemplate.from_template(
    """Eres un agente de acceso a documentos. Buscas y presentas información factual.
Responde basándote SOLO en el contexto recuperado.

Contexto recuperado:
{contexto}

Pregunta: {pregunta}
Respuesta:"""
)


# ── Nodos ─────────────────────────────────────────────────────────────────────

def nodo_general(estado: EstadoHandoff) -> Command:
    """
    El agente general evalúa la pregunta y decide:
    - Responder él mismo, o
    - Hacer handoff a un agente especializado
    """
    llm = get_llm()
    pregunta = estado["messages"][-1].content
    historial = "\n".join(
        f"{type(m).__name__}: {m.content[:60]}"
        for m in estado["messages"][:-1]
    ) or "ninguno"

    respuesta = (PROMPT_GENERAL | llm).invoke({
        "pregunta": pregunta,
        "mensajes": historial,
    }).content.strip()

    console.print(f"  [yellow]agente_general:[/] '{respuesta[:60]}'")

    # Si el agente decide hacer handoff
    if respuesta.startswith("HANDOFF:"):
        destino = respuesta.split(":")[1].strip()
        console.print(f"  [bold yellow]→ HANDOFF a {destino}[/]")
        return Command(
            goto=destino,
            update={
                "agente_actual": destino,
                "historial_handoffs": [f"general → {destino}"],
            },
        )

    # Respuesta directa
    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content=respuesta, name="agente_general")],
            "agente_actual": "general",
        },
    )


def nodo_tecnico(estado: EstadoHandoff) -> Command:
    """Agente técnico especializado en LangChain/LangGraph."""
    llm = get_llm()
    pregunta = estado["messages"][-1].content

    respuesta = (PROMPT_TECNICO | llm).invoke({
        "pregunta": pregunta,
        "contexto": "LangGraph 1.1.3, LangChain 1.2, Python 3.10+",
    }).content.strip()

    console.print(f"  [cyan]agente_tecnico:[/] '{respuesta[:60]}'")

    if respuesta.startswith("HANDOFF:"):
        destino = respuesta.split(":")[1].strip()
        console.print(f"  [bold yellow]→ HANDOFF a {destino}[/]")
        return Command(
            goto=destino,
            update={
                "agente_actual": destino,
                "historial_handoffs": [f"tecnico → {destino}"],
            },
        )

    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content=respuesta, name="agente_tecnico")],
            "agente_actual": "tecnico",
        },
    )


def nodo_rag(estado: EstadoHandoff) -> Command:
    """Agente RAG: busca en documentos y responde."""
    pregunta = estado["messages"][-1].content
    retriever = get_retriever()
    docs = retriever.invoke(pregunta)
    contexto = format_docs(docs[:2])

    llm = get_llm()
    respuesta = (PROMPT_RAG | llm).invoke({
        "pregunta": pregunta,
        "contexto": contexto,
    }).content

    console.print(f"  [green]agente_rag:[/] {len(docs)} docs, {len(respuesta)} chars")

    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content=respuesta, name="agente_rag")],
            "agente_actual": "rag",
            "historial_handoffs": [f"rag respondió"],
        },
    )


# ── Grafo ─────────────────────────────────────────────────────────────────────

def construir_grafo():
    builder = StateGraph(EstadoHandoff)

    builder.add_node("agente_general", nodo_general)
    builder.add_node("agente_tecnico", nodo_tecnico)
    builder.add_node("agente_rag", nodo_rag)

    builder.add_edge(START, "agente_general")
    # No hay add_conditional_edges aquí — el routing lo hacen
    # los propios nodos con Command(goto="destino")

    return builder.compile(checkpointer=MemorySaver())


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 11.3: Handoff")

    console.print(Panel(
        "[bold]Handoff vs Supervisor:[/]\n\n"
        "  [cyan]Supervisor[/]: agente central que orquesta (top-down)\n"
        "  [cyan]Handoff[/]:    el agente activo decide pasar el control (peer-to-peer)\n\n"
        "[bold]Implementación:[/]\n"
        "[dim]  # En vez de retornar un dict, el nodo retorna un Command:\n"
        "  return Command(\n"
        "      goto='agente_tecnico',     # próximo nodo\n"
        "      update={'agente_actual': 'tecnico'},  # cambios al estado\n"
        "  )[/]\n\n"
        "El grafo salta directamente al nodo indicado en goto.",
        border_style="blue",
        title="Handoff con Command(goto=...)",
    ))

    grafo = construir_grafo()

    console.print(Panel(
        grafo.get_graph().draw_ascii(),
        title="Grafo de handoff",
        border_style="yellow",
    ))

    preguntas = [
        ("Hola, ¿me puedes ayudar?", "esperado: general responde directo"),
        ("¿Cómo funciona StateGraph en LangGraph 1.1?", "esperado: handoff → técnico"),
        ("¿De qué trata el documento cargado?", "esperado: handoff → rag"),
    ]

    for pregunta, nota in preguntas:
        console.rule(f"[bold]{pregunta}")
        console.print(f"  [dim]{nota}[/]")

        cfg = {"configurable": {"thread_id": f"handoff-{hash(pregunta) % 9999}"}}
        estado_inicial: EstadoHandoff = {
            "messages": [HumanMessage(content=pregunta)],
            "agente_actual": "general",
            "historial_handoffs": [],
        }
        resultado = grafo.invoke(estado_inicial, config=cfg)

        respuesta_final = next(
            (m.content for m in reversed(resultado["messages"]) if isinstance(m, AIMessage)),
            "sin respuesta",
        )
        console.print(Panel(
            f"[dim]Atendido por: {resultado['agente_actual']}[/]\n"
            f"[dim]Handoffs: {resultado['historial_handoffs'] or ['ninguno']}[/]\n\n"
            + respuesta_final[:300],
            title="Resultado",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
