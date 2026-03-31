"""
56_subgraphs.py — Módulo 11.2: Subgraphs — grafos dentro de grafos

Un subgraph es un grafo compilado que se usa como nodo dentro de otro grafo.
Esto permite modularizar sistemas complejos: cada subgraph es una unidad
independiente con su propio estado interno.

El estado del subgraph se "mapea" al estado del grafo padre:
  - Las keys que comparten mismo nombre se pasan automáticamente
  - Puedes transformar las keys con funciones de mapeo explícitas

Casos de uso:
  - Encapsular el Corrective RAG del módulo 8 como un nodo reutilizable
  - Separar la lógica de búsqueda de la lógica de generación
  - Reutilizar subgraphs en múltiples grafos padre

Este script convierte el pipeline RAG en un subgraph y lo usa
desde un grafo padre que añade enrutamiento y post-procesamiento.
"""
from typing import TypedDict, Annotated
import operator

from rich.console import Console
from rich.panel import Panel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Estado del SUBGRAPH (pipeline RAG interno) ────────────────────────────────

class EstadoRAGInterno(TypedDict):
    pregunta: str
    documentos: list[Document]
    contexto: str
    respuesta: str


# ── Nodos del subgraph ────────────────────────────────────────────────────────

def sub_retriever(estado: EstadoRAGInterno) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])
    console.print(f"    [cyan][sub] retriever:[/] {len(docs)} docs")
    return {"documentos": docs}


def sub_generator(estado: EstadoRAGInterno) -> dict:
    contexto = format_docs(estado["documentos"])
    llm = get_llm()
    respuesta = (QUERY_PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content
    console.print(f"    [green][sub] generator:[/] {len(respuesta)} chars")
    return {"contexto": contexto, "respuesta": respuesta}


# ── Construir el subgraph ─────────────────────────────────────────────────────

def construir_subgraph_rag():
    """
    Este grafo se compilará y usará como NODO dentro del grafo padre.
    No necesita checkpointer propio (hereda el del padre si lo hay).
    """
    builder = StateGraph(EstadoRAGInterno)
    builder.add_node("retriever", sub_retriever)
    builder.add_node("generator", sub_generator)
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)
    return builder.compile()


# ── Estado del grafo PADRE ────────────────────────────────────────────────────

class EstadoPadre(TypedDict):
    pregunta: str           # compartida con el subgraph (mismo nombre)
    tipo_pregunta: str
    respuesta: str          # compartida con el subgraph (mismo nombre)
    respuesta_formateada: str
    logs: Annotated[list[str], operator.add]


# ── Nodos del grafo padre ─────────────────────────────────────────────────────

def nodo_clasificador(estado: EstadoPadre) -> dict:
    """Clasifica la pregunta antes de pasarla al subgraph."""
    pregunta = estado["pregunta"].lower()
    tipo = "documental" if any(w in pregunta for w in ["documento", "texto", "dice", "trata"]) else "general"
    console.print(f"  [yellow]padre → clasificador:[/] tipo={tipo}")
    return {
        "tipo_pregunta": tipo,
        "logs": [f"clasificador: tipo={tipo}"],
    }


# El subgraph se usará como nodo directamente — LangGraph mapea las keys
# que tienen el mismo nombre en ambos estados (pregunta, respuesta)

def nodo_formateador(estado: EstadoPadre) -> dict:
    """Post-procesa la respuesta del subgraph."""
    respuesta = estado.get("respuesta", "")
    formateada = f"[Respuesta {estado['tipo_pregunta'].upper()}]\n\n{respuesta}"
    console.print(f"  [magenta]padre → formateador:[/] añadiendo prefijo")
    return {
        "respuesta_formateada": formateada,
        "logs": [f"formateador: añadió prefijo '{estado['tipo_pregunta']}'"],
    }


# ── Grafo padre ───────────────────────────────────────────────────────────────

def construir_grafo_padre():
    subgraph_rag = construir_subgraph_rag()

    builder = StateGraph(EstadoPadre)

    builder.add_node("clasificador", nodo_clasificador)
    # El subgraph compilado se usa directamente como nodo
    # LangGraph mapea automáticamente las keys con mismo nombre
    builder.add_node("rag", subgraph_rag)
    builder.add_node("formateador", nodo_formateador)

    builder.add_edge(START, "clasificador")
    builder.add_edge("clasificador", "rag")
    builder.add_edge("rag", "formateador")
    builder.add_edge("formateador", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]RAG Lab — Módulo 11.2: Subgraphs")

    console.print(Panel(
        "[bold]Subgraph = grafo compilado usado como nodo[/]\n\n"
        "  Grafo padre:                    Subgraph RAG:\n"
        "  START → clasificador            START → retriever\n"
        "             ↓                               ↓\n"
        "          [bold cyan]nodo 'rag'[/]                  generator\n"
        "          (es el subgraph)              ↓\n"
        "             ↓                         END\n"
        "         formateador\n"
        "             ↓\n"
        "            END\n\n"
        "[dim]El subgraph tiene su propio estado interno (EstadoRAGInterno).\n"
        "Las keys 'pregunta' y 'respuesta' se mapean automáticamente\n"
        "porque tienen el mismo nombre en ambos estados.[/]",
        border_style="blue",
        title="Subgraphs",
    ))

    # Mostrar ambos grafos
    subgraph = construir_subgraph_rag()
    grafo_padre = construir_grafo_padre()

    console.print(Panel(
        subgraph.get_graph().draw_ascii(),
        title="Subgraph RAG (interno)",
        border_style="cyan",
    ))
    console.print(Panel(
        grafo_padre.get_graph(xray=True).draw_ascii(),
        title="Grafo padre con subgraph expandido (xray=True)",
        border_style="yellow",
    ))

    preguntas = [
        "¿De qué trata el documento?",
        "¿Cuáles son los temas principales del texto?",
    ]

    for pregunta in preguntas:
        console.rule(f"[bold]{pregunta}")

        estado_inicial: EstadoPadre = {
            "pregunta": pregunta,
            "tipo_pregunta": "",
            "respuesta": "",
            "respuesta_formateada": "",
            "logs": [],
        }

        resultado = grafo_padre.invoke(estado_inicial)

        console.print(Panel(
            resultado["respuesta_formateada"][:400],
            title="Respuesta formateada",
            border_style="green",
        ))
        console.print(f"  [dim]Logs: {resultado['logs']}[/]")

    # Ventajas de los subgraphs
    console.print(Panel(
        "[bold]Ventajas de los subgraphs:[/]\n\n"
        "  [cyan]Modularidad[/]:   cada subgraph es una unidad independiente y testeable\n"
        "  [cyan]Reutilización[/]: el mismo subgraph_rag puede usarse en múltiples grafos padre\n"
        "  [cyan]Encapsulación[/]: el estado interno del subgraph no contamina el padre\n"
        "  [cyan]xray=True[/]:     `get_graph(xray=True)` expande los subgraphs para visualización",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
