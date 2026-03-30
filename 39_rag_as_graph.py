"""
39_rag_as_graph.py — Módulo 8.1: Migrar el RAG actual de LCEL a LangGraph

El RAG en LCEL es un pipe lineal:
  retriever | format_docs | prompt | llm | StrOutputParser

En LangGraph se convierte en un grafo con nodos discretos:
  START → retriever → generator → END

¿Por qué migrar?
  - Cada paso es inspectable por separado
  - Puedes insertar nodos de validación entre pasos
  - Puedes agregar ciclos (reformulación) sin romper la estructura
  - El estado centraliza todos los datos del flujo
"""
from typing import TypedDict, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Estado del RAG ────────────────────────────────────────────────────────────
# Contiene todo lo que los nodos necesitan compartir.

class EstadoRAG(TypedDict):
    pregunta: str
    documentos: list[Document]
    contexto: str
    respuesta: str
    logs: Annotated[list[str], operator.add]


# ── Nodos ─────────────────────────────────────────────────────────────────────

def nodo_retriever(estado: EstadoRAG) -> dict:
    """Busca documentos relevantes para la pregunta."""
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])

    console.print(f"  [cyan]retriever:[/] {len(docs)} documentos recuperados")

    return {
        "documentos": docs,
        "logs": [f"retriever: {len(docs)} docs para '{estado['pregunta'][:40]}'"],
    }


def nodo_generator(estado: EstadoRAG) -> dict:
    """Genera la respuesta usando los documentos recuperados."""
    contexto = format_docs(estado["documentos"])
    llm = get_llm()

    chain = QUERY_PROMPT | llm
    respuesta = chain.invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content

    console.print(f"  [green]generator:[/] {len(respuesta)} caracteres generados")

    return {
        "contexto": contexto,
        "respuesta": respuesta,
        "logs": [f"generator: respuesta generada ({len(respuesta)} chars)"],
    }


# ── Grafo ─────────────────────────────────────────────────────────────────────

def construir_rag_graph():
    builder = StateGraph(EstadoRAG)

    builder.add_node("retriever", nodo_retriever)
    builder.add_node("generator", nodo_generator)

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]RAG Lab — Módulo 8.1: RAG como Grafo")

    console.print(Panel(
        "[bold]LCEL (antes):[/]\n"
        "[dim]retriever | format_docs | prompt | llm | StrOutputParser[/]\n\n"
        "[bold]LangGraph (ahora):[/]\n"
        "[dim]START → nodo_retriever → nodo_generator → END[/]\n\n"
        "Misma lógica, estructura diferente:\n"
        "  • Cada paso es una función testeable independientemente\n"
        "  • El estado centraliza los datos (docs, contexto, respuesta)\n"
        "  • Fácil agregar nodos de validación entre retriever y generator",
        border_style="blue",
        title="LCEL → LangGraph",
    ))

    grafo = construir_rag_graph()

    # Mostrar estructura
    console.print(Panel(
        grafo.get_graph().draw_ascii(),
        title="Estructura del grafo",
        border_style="yellow",
    ))

    pregunta = "¿De qué trata el documento?"
    console.print(f"\n[bold]Pregunta:[/] {pregunta}")
    console.print("[bold]Ejecutando...[/]\n")

    estado_inicial: EstadoRAG = {
        "pregunta": pregunta,
        "documentos": [],
        "contexto": "",
        "respuesta": "",
        "logs": [],
    }

    resultado = grafo.invoke(estado_inicial)

    console.print(Panel(
        resultado["respuesta"],
        title="Respuesta",
        border_style="green",
    ))

    # Comparación directa
    tabla = Table(title="Qué cambia y qué no", show_header=True, header_style="bold magenta")
    tabla.add_column("Aspecto", style="bold")
    tabla.add_column("LCEL")
    tabla.add_column("LangGraph")

    tabla.add_row("Retriever", "retriever.invoke()", "nodo_retriever(estado)")
    tabla.add_row("Contexto", "format_docs() en el pipe", "nodo_generator accede a estado['documentos']")
    tabla.add_row("Prompt", "QUERY_PROMPT | llm", "igual, dentro del nodo")
    tabla.add_row("Datos intermedios", "No accesibles", "estado['documentos'], estado['contexto']")
    tabla.add_row("Testing", "Difícil aislar pasos", "Cada nodo testeable por separado")

    console.print(tabla)

    console.print(Panel(
        "\n".join(f"  • {log}" for log in resultado["logs"]),
        title="Logs del flujo",
        border_style="dim",
    ))


if __name__ == "__main__":
    main()
