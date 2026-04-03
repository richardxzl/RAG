"""
58_shared_vs_isolated_state.py — Módulo 11.4: Shared state vs Isolated state

En sistemas multi-agente hay dos modelos de estado:

  SHARED STATE:   todos los agentes leen y escriben el mismo estado
    + Simple de implementar (un solo TypedDict)
    + Los agentes pueden ver el trabajo de otros
    - Riesgo de colisiones (dos agentes escribiendo la misma key)
    - El estado puede crecer demasiado

  ISOLATED STATE: cada agente tiene su propio estado privado
    + Sin colisiones
    + Estado limpio por agente
    - Más código de plumbing para pasar resultados entre agentes
    - Necesitas decidir qué compartir explícitamente

LangGraph ofrece Send() para el patrón "fan-out / fan-in":
  1. Fan-out:  desde un nodo, despachar trabajo a N agentes en paralelo (Send)
  2. Fan-in:   recoger todos los resultados en un nodo aggregador
"""
import logging
from typing import TypedDict, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from rag.chain import get_llm, format_docs
from rag.retriever import get_retriever

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# PATRÓN 1: SHARED STATE — todos los agentes comparten el mismo estado
# ══════════════════════════════════════════════════════════════════════════════

class EstadoCompartido(TypedDict):
    pregunta: str
    # Cada agente escribe en su propia key → sin colisiones
    resultado_rag: str
    resultado_analisis: str
    resultado_formato: str
    respuesta_final: str
    logs: Annotated[list[str], operator.add]


def agente_rag_shared(estado: EstadoCompartido) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])
    contexto = format_docs(docs[:2])
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Resume brevemente la información relevante para: {pregunta}\nContexto: {contexto}"
    )
    resultado = (prompt | llm).invoke({"pregunta": estado["pregunta"], "contexto": contexto}).content
    console.print(f"  [cyan]shared→rag:[/] {len(resultado)} chars")
    return {
        "resultado_rag": resultado,
        "logs": ["rag completado"],
    }


def agente_analisis_shared(estado: EstadoCompartido) -> dict:
    """Lee resultado_rag del estado compartido para construir sobre él."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Analiza críticamente esta información:\n{info}\n\nPuntos clave:"
    )
    resultado = (prompt | llm).invoke({"info": estado["resultado_rag"] or "sin info previa"}).content
    console.print(f"  [magenta]shared→analisis:[/] {len(resultado)} chars")
    return {
        "resultado_analisis": resultado,
        "logs": ["analisis completado"],
    }


def agente_formato_shared(estado: EstadoCompartido) -> dict:
    """Lee los resultados de los otros dos agentes y genera la respuesta final."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Combina esta información en una respuesta clara:\n"
        "Info RAG: {rag}\nAnálisis: {analisis}\n\nRespuesta final:"
    )
    resultado = (prompt | llm).invoke({
        "rag": estado["resultado_rag"],
        "analisis": estado["resultado_analisis"],
    }).content
    console.print(f"  [green]shared→formato:[/] {len(resultado)} chars")
    return {
        "resultado_formato": resultado,
        "respuesta_final": resultado,
        "logs": ["formato completado"],
    }


def construir_grafo_shared():
    """
    Shared state: los nodos se ejecutan en secuencia,
    cada uno lee el resultado del anterior del estado compartido.
    """
    builder = StateGraph(EstadoCompartido)
    builder.add_node("rag", agente_rag_shared)
    builder.add_node("analisis", agente_analisis_shared)
    builder.add_node("formato", agente_formato_shared)
    builder.add_edge(START, "rag")
    builder.add_edge("rag", "analisis")
    builder.add_edge("analisis", "formato")
    builder.add_edge("formato", END)
    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# PATRÓN 2: ISOLATED STATE con fan-out / fan-in usando Send()
# ══════════════════════════════════════════════════════════════════════════════

class EstadoPadreIsolated(TypedDict):
    pregunta: str
    perspectivas: list[str]        # roles a explorar en paralelo
    resultados: Annotated[list[str], operator.add]  # acumula resultados de cada agente
    respuesta_final: str


class EstadoAgenteIsolated(TypedDict):
    """Estado privado de cada agente — invisible para los demás."""
    pregunta: str
    perspectiva: str    # el rol específico de este agente
    resultado: str


def nodo_fan_out(estado: EstadoPadreIsolated) -> list[Send]:
    """
    Fan-out: despacha el mismo trabajo a múltiples agentes en paralelo.
    Cada Send crea una instancia independiente del nodo destino con
    su propio estado aislado.
    """
    console.print(f"  [yellow]fan-out:[/] {len(estado['perspectivas'])} agentes en paralelo")
    return [
        Send("agente_perspectiva", {
            "pregunta": estado["pregunta"],
            "perspectiva": perspectiva,
            "resultado": "",
        })
        for perspectiva in estado["perspectivas"]
    ]


def nodo_agente_perspectiva(estado: EstadoAgenteIsolated) -> dict:
    """
    Cada instancia de este nodo tiene su propio estado aislado.
    No puede leer los resultados de las otras instancias.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Responde desde la perspectiva de un {perspectiva}.\n"
        "Sé breve (2-3 oraciones).\n\nPregunta: {pregunta}"
    )
    resultado = (prompt | llm).invoke({
        "perspectiva": estado["perspectiva"],
        "pregunta": estado["pregunta"],
    }).content
    console.print(f"  [cyan]agente ({estado['perspectiva'][:20]}):[/] {len(resultado)} chars")
    # Los resultados se acumulan en el estado padre gracias al reducer operator.add
    return {"resultados": [f"[{estado['perspectiva']}]: {resultado}"]}


def nodo_fan_in(estado: EstadoPadreIsolated) -> dict:
    """Fan-in: agrega todos los resultados en una respuesta final."""
    llm = get_llm()
    todos = "\n\n".join(estado["resultados"])
    prompt = ChatPromptTemplate.from_template(
        "Sintetiza estas perspectivas en una respuesta equilibrada:\n{perspectivas}"
    )
    respuesta = (prompt | llm).invoke({"perspectivas": todos}).content
    console.print(f"  [green]fan-in:[/] {len(estado['resultados'])} perspectivas sintetizadas")
    return {"respuesta_final": respuesta}


def construir_grafo_isolated():
    """
    Isolated state con fan-out / fan-in:
    - nodo_fan_out despacha N instancias de agente_perspectiva en PARALELO
    - Cada instancia tiene su propio estado (EstadoAgenteIsolated)
    - Sus resultados se acumulan en estado padre vía operator.add
    - nodo_fan_in agrega todo
    """
    builder = StateGraph(EstadoPadreIsolated)
    builder.add_node("agente_perspectiva", nodo_agente_perspectiva)
    builder.add_node("fan_in", nodo_fan_in)

    # Fan-out desde START
    builder.add_conditional_edges(START, nodo_fan_out, ["agente_perspectiva"])
    builder.add_edge("agente_perspectiva", "fan_in")
    builder.add_edge("fan_in", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 11.4: Shared vs Isolated State")

    console.print(Panel(
        "[bold]Shared State:[/]  un TypedDict, todos los nodos leen/escriben en él\n"
        "  Bueno para: pipelines secuenciales donde cada paso construye sobre el anterior\n\n"
        "[bold]Isolated State:[/] cada agente tiene su propio estado privado\n"
        "  Bueno para: trabajo paralelo independiente (fan-out / fan-in)\n\n"
        "[bold]Send():[/] mecanismo para crear instancias de nodos en paralelo\n"
        "[dim]  return [Send('nodo', estado_privado_1), Send('nodo', estado_privado_2)][/]",
        border_style="blue",
        title="Shared vs Isolated State",
    ))

    pregunta = "¿Cuáles son los principales desafíos de los sistemas RAG?"

    # Demo 1: Shared state (secuencial)
    console.rule("[yellow]Patrón 1: Shared State (secuencial)")
    grafo_shared = construir_grafo_shared()
    console.print(Panel(grafo_shared.get_graph().draw_ascii(), title="Grafo shared", border_style="cyan"))

    resultado_shared = grafo_shared.invoke({
        "pregunta": pregunta,
        "resultado_rag": "",
        "resultado_analisis": "",
        "resultado_formato": "",
        "respuesta_final": "",
        "logs": [],
    })
    console.print(Panel(resultado_shared["respuesta_final"][:300], title="Resultado shared", border_style="green"))

    # Demo 2: Isolated state (paralelo con Send)
    console.rule("[yellow]Patrón 2: Isolated State + Fan-out/Fan-in (paralelo)")
    grafo_isolated = construir_grafo_isolated()
    console.print(Panel(grafo_isolated.get_graph().draw_ascii(), title="Grafo isolated", border_style="cyan"))

    resultado_isolated = grafo_isolated.invoke({
        "pregunta": pregunta,
        "perspectivas": ["ingeniero de software", "investigador académico", "product manager"],
        "resultados": [],
        "respuesta_final": "",
    })
    console.print(Panel(resultado_isolated["respuesta_final"][:300], title="Resultado isolated (3 perspectivas)", border_style="green"))

    # Comparativa
    tabla = Table(title="Shared vs Isolated", show_header=True, header_style="bold magenta")
    tabla.add_column("Aspecto", style="bold")
    tabla.add_column("Shared State")
    tabla.add_column("Isolated State")
    tabla.add_row("Ejecución", "Secuencial", "Paralela (Send)")
    tabla.add_row("Visibilidad", "Todos ven todo", "Solo el propio estado")
    tabla.add_row("Colisiones", "Posibles (mismo nombre)", "Imposibles")
    tabla.add_row("Complejidad", "Baja", "Media (fan-out/fan-in)")
    tabla.add_row("Escalabilidad", "Limitada", "Alta (N agentes en paralelo)")
    tabla.add_row("Cuándo usar", "Pipeline secuencial", "Trabajo independiente paralelo")
    console.print(tabla)


if __name__ == "__main__":
    main()
