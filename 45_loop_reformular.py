"""
45_loop_reformular.py — Módulo 8.7: Loop — reformular → reintentar (max 2 veces)

Este script implementa el patrón completo de "Corrective RAG":
  1. Router:    ¿necesita RAG o es conversacional?
  2. Retriever: busca chunks
  3. Grader:    ¿los chunks son relevantes?
     → No relevantes: reformula la pregunta → vuelve al retriever (max 2 veces)
  4. Generator: genera respuesta con chunks relevantes
  5. Hallucination check: ¿la respuesta es fiel a los chunks?
     → No fiel: regenera (max 2 veces)

Este es el grafo más completo del módulo 8. Cada nodo de los módulos anteriores
se conecta aquí como piezas de un sistema coherente.
"""
import logging
from typing import TypedDict, Literal, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()

MAX_REFORMULACIONES = 2
MAX_REGENERACIONES = 2


# ── Estado completo ───────────────────────────────────────────────────────────

class EstadoCorrectiveRAG(TypedDict):
    pregunta_original: str
    pregunta_actual: str        # puede cambiar en cada reformulación
    ruta: str                   # "rag" o "conversacional"
    documentos: list[Document]
    documentos_relevantes: list[Document]
    contexto: str
    respuesta: str
    es_fiel: bool
    reformulaciones: int
    regeneraciones: int
    logs: Annotated[list[str], operator.add]


# ── Prompts ───────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """Clasifica la pregunta. Responde SOLO: rag o conversacional.
Pregunta: {pregunta}
Respuesta:"""
)

GRADER_PROMPT = ChatPromptTemplate.from_template(
    """¿Este fragmento ayuda a responder la pregunta?
Pregunta: {pregunta}
Fragmento: {fragmento}
Responde SOLO: relevante o irrelevante"""
)

REFORMULAR_PROMPT = ChatPromptTemplate.from_template(
    """La pregunta original no encontró documentos relevantes.
Reformúlala para mejorar la búsqueda. Mantén el mismo significado pero usa otras palabras.

Pregunta original: {pregunta}
Intento número: {intento}

Nueva pregunta reformulada (SOLO la pregunta, sin explicación):"""
)

HALLUCINATION_PROMPT = ChatPromptTemplate.from_template(
    """¿La respuesta está completamente respaldada por el contexto?
Contexto: {contexto}
Respuesta: {respuesta}
Responde SOLO: fiel o alucinacion"""
)


# ── Nodos ─────────────────────────────────────────────────────────────────────

def nodo_router(estado: EstadoCorrectiveRAG) -> dict:
    llm = get_llm()
    resultado = (ROUTER_PROMPT | llm).invoke({
        "pregunta": estado["pregunta_actual"]
    }).content.strip().lower()
    ruta = "rag" if "rag" in resultado else "conversacional"
    console.print(f"  [yellow]router:[/] → {ruta}")
    return {"ruta": ruta, "logs": [f"router: {ruta}"]}


def nodo_retriever(estado: EstadoCorrectiveRAG) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta_actual"])
    console.print(f"  [cyan]retriever:[/] {len(docs)} docs para '{estado['pregunta_actual'][:40]}'")
    return {
        "documentos": docs,
        "logs": [f"retriever[{estado['reformulaciones']}]: {len(docs)} docs"],
    }


def nodo_grader(estado: EstadoCorrectiveRAG) -> dict:
    llm = get_llm()
    cadena = GRADER_PROMPT | llm
    relevantes = []
    for doc in estado["documentos"]:
        veredicto = cadena.invoke({
            "pregunta": estado["pregunta_actual"],
            "fragmento": doc.page_content[:400],
        }).content.strip().lower()
        if "relevante" in veredicto and "irrelevante" not in veredicto:
            relevantes.append(doc)

    console.print(f"  [cyan]grader:[/] {len(relevantes)}/{len(estado['documentos'])} relevantes")
    return {
        "documentos_relevantes": relevantes,
        "logs": [f"grader[{estado['reformulaciones']}]: {len(relevantes)}/{len(estado['documentos'])} relevantes"],
    }


def nodo_reformular(estado: EstadoCorrectiveRAG) -> dict:
    """Reformula la pregunta cuando el grader no encontró docs relevantes."""
    llm = get_llm()
    nueva_pregunta = (REFORMULAR_PROMPT | llm).invoke({
        "pregunta": estado["pregunta_actual"],
        "intento": estado["reformulaciones"] + 1,
    }).content.strip()

    console.print(f"  [magenta]reformular:[/] '{estado['pregunta_actual'][:40]}' → '{nueva_pregunta[:40]}'")
    return {
        "pregunta_actual": nueva_pregunta,
        "reformulaciones": estado["reformulaciones"] + 1,
        "documentos": [],
        "documentos_relevantes": [],
        "logs": [f"reformular[{estado['reformulaciones'] + 1}]: nueva pregunta='{nueva_pregunta[:50]}'"],
    }


def nodo_generator(estado: EstadoCorrectiveRAG) -> dict:
    docs = estado["documentos_relevantes"] or estado["documentos"]
    contexto = format_docs(docs)
    llm = get_llm()
    respuesta = (QUERY_PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["pregunta_original"],  # siempre respondemos la original
    }).content
    console.print(f"  [green]generator:[/] {len(respuesta)} chars")
    return {
        "contexto": contexto,
        "respuesta": respuesta,
        "regeneraciones": estado["regeneraciones"] + 1,
        "logs": [f"generator[{estado['regeneraciones'] + 1}]: {len(respuesta)} chars"],
    }


def nodo_directo(estado: EstadoCorrectiveRAG) -> dict:
    llm = get_llm()
    respuesta = llm.invoke([HumanMessage(content=estado["pregunta_actual"])]).content
    return {
        "respuesta": respuesta,
        "es_fiel": True,  # respuesta directa, no aplica hallucination check
        "logs": ["directo: respuesta conversacional"],
    }


def nodo_hallucination_check(estado: EstadoCorrectiveRAG) -> dict:
    llm = get_llm()
    veredicto = (HALLUCINATION_PROMPT | llm).invoke({
        "contexto": estado["contexto"][:1500],
        "respuesta": estado["respuesta"],
    }).content.strip().lower()
    es_fiel = "alucinacion" not in veredicto
    console.print(f"  [cyan]hallucination_check:[/] {'[green]fiel[/]' if es_fiel else '[red]alucinación[/]'}")
    return {
        "es_fiel": es_fiel,
        "logs": [f"hallucination_check[{estado['regeneraciones']}]: fiel={es_fiel}"],
    }


# ── Funciones de routing (edges condicionales) ────────────────────────────────

def router_decision(estado: EstadoCorrectiveRAG) -> Literal["retriever", "directo"]:
    return "retriever" if estado["ruta"] == "rag" else "directo"


def grader_decision(
    estado: EstadoCorrectiveRAG,
) -> Literal["generator", "reformular"]:
    if estado["documentos_relevantes"]:
        return "generator"
    if estado["reformulaciones"] >= MAX_REFORMULACIONES:
        console.print(
            f"  [yellow]Máximo de reformulaciones ({MAX_REFORMULACIONES}) → generando de todos modos[/]"
        )
        return "generator"
    return "reformular"


def hallucination_decision(
    estado: EstadoCorrectiveRAG,
) -> Literal["generator", "__end__"]:
    if estado["es_fiel"]:
        return END
    if estado["regeneraciones"] >= MAX_REGENERACIONES:
        console.print(f"  [yellow]Máximo de regeneraciones → finalizando[/]")
        return END
    return "generator"


# ── Grafo completo ────────────────────────────────────────────────────────────

def construir_grafo_corrective_rag():
    builder = StateGraph(EstadoCorrectiveRAG)

    builder.add_node("router", nodo_router)
    builder.add_node("retriever", nodo_retriever)
    builder.add_node("grader", nodo_grader)
    builder.add_node("reformular", nodo_reformular)
    builder.add_node("generator", nodo_generator)
    builder.add_node("directo", nodo_directo)
    builder.add_node("hallucination_check", nodo_hallucination_check)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", router_decision, {
        "retriever": "retriever",
        "directo": "directo",
    })
    builder.add_edge("retriever", "grader")
    builder.add_conditional_edges("grader", grader_decision, {
        "generator": "generator",
        "reformular": "reformular",
    })
    builder.add_edge("reformular", "retriever")   # ← el loop de reformulación
    builder.add_edge("generator", "hallucination_check")
    builder.add_conditional_edges("hallucination_check", hallucination_decision, {
        "generator": "generator",                  # ← el loop de regeneración
        END: END,
    })
    builder.add_edge("directo", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 8.7: Corrective RAG completo")

    grafo = construir_grafo_corrective_rag()

    console.print(Panel(
        grafo.get_graph().draw_ascii(),
        title="Grafo Corrective RAG completo",
        border_style="yellow",
    ))

    console.print(Panel(
        "[bold]Flujo completo:[/]\n\n"
        "  1. [yellow]Router[/]:              ¿RAG o conversacional?\n"
        "  2. [cyan]Retriever[/]:          busca chunks\n"
        "  3. [cyan]Grader[/]:             ¿son relevantes?\n"
        "     └─ No → [magenta]Reformular[/] (max 2x) → vuelve a Retriever\n"
        "  4. [green]Generator[/]:         genera respuesta\n"
        "  5. [cyan]Hallucination Check[/]: ¿es fiel?\n"
        "     └─ No → [green]Regenerar[/] (max 2x)",
        border_style="blue",
        title="Corrective RAG",
    ))

    preguntas = [
        "¿De qué trata el documento?",
        "Hola, ¿puedes ayudarme?",
        "¿Cuáles son los temas principales del texto?",
    ]

    for pregunta in preguntas:
        console.rule(f"[bold]Pregunta: {pregunta}")

        estado: EstadoCorrectiveRAG = {
            "pregunta_original": pregunta,
            "pregunta_actual": pregunta,
            "ruta": "",
            "documentos": [],
            "documentos_relevantes": [],
            "contexto": "",
            "respuesta": "",
            "es_fiel": False,
            "reformulaciones": 0,
            "regeneraciones": 0,
            "logs": [],
        }

        resultado = grafo.invoke(estado)

        color = "green" if resultado["es_fiel"] or resultado["ruta"] == "conversacional" else "yellow"
        console.print(Panel(
            resultado["respuesta"][:300] + "...",
            title=f"Respuesta | ruta={resultado['ruta']} | reformulaciones={resultado['reformulaciones']} | regeneraciones={resultado['regeneraciones']}",
            border_style=color,
        ))

    # Resumen de qué hicimos en el módulo 8
    tabla = Table(
        title="Módulo 8 — Nodos implementados",
        show_header=True,
        header_style="bold magenta",
    )
    tabla.add_column("Script")
    tabla.add_column("Nodo")
    tabla.add_column("Responsabilidad")

    tabla.add_row("39", "—", "Migrar LCEL → LangGraph (retriever + generator)")
    tabla.add_row("40", "Router", "¿RAG o conversacional? (keywords o LLM)")
    tabla.add_row("41", "Retriever", "Busca chunks + expone scores")
    tabla.add_row("42", "Grader", "¿Chunks relevantes? (score o LLM)")
    tabla.add_row("43", "Generator", "Genera respuesta (básico / fuentes / confianza)")
    tabla.add_row("44", "Hallucination Check", "¿Respuesta fiel al contexto?")
    tabla.add_row("45", "Loop completo", "Corrective RAG: todo integrado con reformulación")

    console.print(tabla)


if __name__ == "__main__":
    main()
