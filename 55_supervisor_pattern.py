"""
55_supervisor_pattern.py — Módulo 11.1: Supervisor pattern

El Supervisor es un agente especial que:
  1. Recibe la tarea original
  2. Decide a cuál sub-agente delegarla (o si puede responder él mismo)
  3. Recibe el resultado del sub-agente
  4. Decide si la tarea está completa o si hay que delegar a otro

Grafo:
  START → supervisor → [agente_rag | agente_calc | agente_resumen] → supervisor → ... → END

El supervisor es el único que puede decidir cuándo terminar.
Los sub-agentes hacen el trabajo y devuelven el control al supervisor.
"""
import logging
from typing import TypedDict, Literal, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()

AGENTES = ["agente_rag", "agente_calculo", "agente_resumen"]

# ── Estado compartido entre supervisor y sub-agentes ─────────────────────────

class EstadoSupervisor(TypedDict):
    tarea_original: str
    mensajes: Annotated[list[str], operator.add]  # historial de delegaciones
    resultado_parcial: str                          # resultado del último sub-agente
    siguiente: str                                  # "agente_X" o "FINISH"
    respuesta_final: str


# ── Prompts ───────────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = ChatPromptTemplate.from_template(
    """Eres un supervisor que coordina un equipo de agentes especializados.

Tarea: {tarea}

Historial de lo realizado:
{historial}

Resultado del último agente: {resultado}

Agentes disponibles:
- agente_rag:     busca información en documentos
- agente_calculo: realiza cálculos y análisis numéricos
- agente_resumen: resume y estructura información

¿Qué hacer ahora?
- Si necesitas información de documentos → agente_rag
- Si necesitas cálculos o números → agente_calculo
- Si ya tienes la información y necesitas estructurarla → agente_resumen
- Si la tarea está completa → FINISH

Responde SOLO con una de estas palabras: agente_rag, agente_calculo, agente_resumen, FINISH"""
)


# ── Nodo Supervisor ───────────────────────────────────────────────────────────

def nodo_supervisor(estado: EstadoSupervisor) -> dict:
    llm = get_llm()
    historial = "\n".join(estado["mensajes"]) if estado["mensajes"] else "ninguno"

    decision = (SUPERVISOR_PROMPT | llm).invoke({
        "tarea": estado["tarea_original"],
        "historial": historial,
        "resultado": estado["resultado_parcial"] or "ninguno",
    }).content.strip().lower()

    # Normalizar la decisión
    siguiente = "FINISH"
    for agente in AGENTES:
        if agente in decision:
            siguiente = agente
            break

    console.print(f"  [yellow]supervisor:[/] → {siguiente}")
    return {
        "siguiente": siguiente,
        "mensajes": [f"supervisor decidió: {siguiente}"],
    }


# ── Sub-agentes ───────────────────────────────────────────────────────────────

def nodo_agente_rag(estado: EstadoSupervisor) -> dict:
    """Busca en los documentos y devuelve el resultado al supervisor."""
    retriever = get_retriever()
    docs = retriever.invoke(estado["tarea_original"])
    contexto = format_docs(docs[:2])
    llm = get_llm()
    respuesta = (QUERY_PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["tarea_original"],
    }).content[:600]

    console.print(f"  [cyan]agente_rag:[/] {len(docs)} docs, {len(respuesta)} chars")
    return {
        "resultado_parcial": respuesta,
        "mensajes": [f"agente_rag completó: {respuesta[:80]}..."],
    }


def nodo_agente_calculo(estado: EstadoSupervisor) -> dict:
    """Analiza numéricamente el resultado parcial o la tarea."""
    import math
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Analiza numéricamente o realiza los cálculos necesarios para esta tarea.\n"
        "Tarea: {tarea}\nContexto disponible: {contexto}\nRespuesta:"
    )
    respuesta = (prompt | llm).invoke({
        "tarea": estado["tarea_original"],
        "contexto": estado["resultado_parcial"] or "sin contexto previo",
    }).content[:400]

    console.print(f"  [magenta]agente_calculo:[/] {len(respuesta)} chars")
    return {
        "resultado_parcial": respuesta,
        "mensajes": [f"agente_calculo completó: {respuesta[:80]}..."],
    }


def nodo_agente_resumen(estado: EstadoSupervisor) -> dict:
    """Resume y estructura toda la información recopilada."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Resume y estructura la siguiente información de forma clara y concisa.\n"
        "Tarea original: {tarea}\n"
        "Información recopilada:\n{info}\n\nResumen estructurado:"
    )
    respuesta = (prompt | llm).invoke({
        "tarea": estado["tarea_original"],
        "info": estado["resultado_parcial"],
    }).content

    console.print(f"  [green]agente_resumen:[/] {len(respuesta)} chars")
    return {
        "resultado_parcial": respuesta,
        "respuesta_final": respuesta,
        "mensajes": [f"agente_resumen completó"],
    }


# ── Routing desde el supervisor ───────────────────────────────────────────────

def routing_supervisor(estado: EstadoSupervisor) -> str:
    sig = estado["siguiente"]
    if sig == "FINISH" or not estado["mensajes"]:
        return END
    return sig


# ── Grafo ─────────────────────────────────────────────────────────────────────

def construir_grafo():
    builder = StateGraph(EstadoSupervisor)

    builder.add_node("supervisor", nodo_supervisor)
    builder.add_node("agente_rag", nodo_agente_rag)
    builder.add_node("agente_calculo", nodo_agente_calculo)
    builder.add_node("agente_resumen", nodo_agente_resumen)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", routing_supervisor, {
        "agente_rag": "agente_rag",
        "agente_calculo": "agente_calculo",
        "agente_resumen": "agente_resumen",
        END: END,
    })

    # Todos los sub-agentes devuelven el control al supervisor
    builder.add_edge("agente_rag", "supervisor")
    builder.add_edge("agente_calculo", "supervisor")
    builder.add_edge("agente_resumen", "supervisor")

    return builder.compile(checkpointer=MemorySaver())


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 11.1: Supervisor Pattern")

    console.print(Panel(
        "[bold]Supervisor pattern:[/]\n\n"
        "  START → [yellow]Supervisor[/] → decide a quién delegar\n"
        "             ↙         ↘\n"
        "    [cyan]agente_rag[/]    [magenta]agente_calculo[/]\n"
        "         ↘         ↙\n"
        "       [green]agente_resumen[/]\n"
        "             ↓\n"
        "       [yellow]Supervisor[/] → ¿terminado? → END\n\n"
        "[dim]El supervisor es el único que puede terminar.\n"
        "Los sub-agentes siempre devuelven el control.[/]",
        border_style="blue",
        title="Supervisor Pattern",
    ))

    grafo = construir_grafo()

    console.print(Panel(
        grafo.get_graph().draw_ascii(),
        title="Grafo del supervisor",
        border_style="yellow",
    ))

    tareas = [
        "¿De qué trata el documento disponible? Dame un resumen estructurado.",
        "¿Cuáles son los temas principales del documento y cuántos hay?",
    ]

    for tarea in tareas:
        console.rule(f"[bold]Tarea: {tarea[:60]}")

        estado_inicial: EstadoSupervisor = {
            "tarea_original": tarea,
            "mensajes": [],
            "resultado_parcial": "",
            "siguiente": "",
            "respuesta_final": "",
        }

        cfg = {"configurable": {"thread_id": f"sup-{hash(tarea) % 9999}"}}
        resultado = grafo.invoke(estado_inicial, config=cfg)

        console.print(Panel(
            f"[dim]Pasos: {len(resultado['mensajes'])}[/]\n\n"
            + (resultado["respuesta_final"] or resultado["resultado_parcial"])[:400],
            title="Resultado final",
            border_style="green",
        ))

        tabla = Table(show_header=True, header_style="bold magenta")
        tabla.add_column("Paso")
        tabla.add_column("Acción")
        for i, msg in enumerate(resultado["mensajes"]):
            tabla.add_row(str(i + 1), msg[:80])
        console.print(tabla)


if __name__ == "__main__":
    main()
