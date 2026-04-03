"""
41_nodo_retriever.py — Módulo 8.3: Nodo Retriever — busca chunks

El nodo retriever en LangGraph es responsable de:
  1. Tomar la pregunta del estado
  2. Ejecutar la búsqueda en el vectorstore
  3. Escribir los documentos recuperados en el estado

A diferencia del retriever en LCEL (que es solo un paso del pipe),
aquí podemos combinar múltiples estrategias y registrar métricas.

Variaciones cubiertas:
  - Retriever simple (similarity search)
  - Retriever con metadata (logging de scores y fuentes)
  - Multi-retriever (pregunta original + reformulada)
"""
import logging
from typing import TypedDict, Annotated
import operator
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever, get_vectorstore
from rag.config import RETRIEVAL_K

console = Console()


# ── Estado ────────────────────────────────────────────────────────────────────

class EstadoRetriever(TypedDict):
    pregunta: str
    documentos: list[Document]
    fuentes: list[str]          # paths de los docs recuperados
    contexto: str
    respuesta: str
    latencia_retriever_ms: float
    logs: Annotated[list[str], operator.add]


# ── Nodo Retriever básico ─────────────────────────────────────────────────────

def nodo_retriever_basico(estado: EstadoRetriever) -> dict:
    """Retriever simple: busca y devuelve los top-k documentos."""
    t0 = time.perf_counter()
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])
    latencia = (time.perf_counter() - t0) * 1000

    fuentes = list({
        doc.metadata.get("source", "desconocido")
        for doc in docs
    })

    console.print(
        f"  [cyan]retriever:[/] {len(docs)} docs en {latencia:.0f}ms "
        f"| fuentes: {fuentes}"
    )

    return {
        "documentos": docs,
        "fuentes": fuentes,
        "latencia_retriever_ms": latencia,
        "logs": [
            f"retriever: {len(docs)} docs, {latencia:.0f}ms, "
            f"fuentes={fuentes}"
        ],
    }


# ── Nodo Retriever con metadata visible ───────────────────────────────────────

def nodo_retriever_con_scores(estado: EstadoRetriever) -> dict:
    """
    Usa similarity_search_with_score para acceder a los scores de similitud.
    Útil para debugging y para el nodo Grader (módulo 8.4).
    """
    t0 = time.perf_counter()
    vs = get_vectorstore()
    resultados = vs.similarity_search_with_score(estado["pregunta"], k=RETRIEVAL_K)
    latencia = (time.perf_counter() - t0) * 1000

    docs = []
    tabla = Table(show_header=True, header_style="bold magenta", title="Documentos recuperados")
    tabla.add_column("#")
    tabla.add_column("Score (similitud)")
    tabla.add_column("Fuente")
    tabla.add_column("Preview (60 chars)")

    for i, (doc, score) in enumerate(resultados):
        # Agregar el score al metadata para que el Grader pueda usarlo
        doc.metadata["retrieval_score"] = float(score)
        docs.append(doc)

        tabla.add_row(
            str(i + 1),
            f"{score:.4f}",
            doc.metadata.get("source", "?")[:30],
            doc.page_content[:60].replace("\n", " "),
        )

    console.print(tabla)

    fuentes = list({d.metadata.get("source", "?") for d in docs})

    return {
        "documentos": docs,
        "fuentes": fuentes,
        "latencia_retriever_ms": latencia,
        "logs": [f"retriever_scores: {len(docs)} docs con scores, {latencia:.0f}ms"],
    }


# ── Nodo Generator (necesario para el grafo completo) ────────────────────────

def nodo_generator(estado: EstadoRetriever) -> dict:
    contexto = format_docs(estado["documentos"])
    llm = get_llm()
    respuesta = (QUERY_PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content
    return {
        "contexto": contexto,
        "respuesta": respuesta,
        "logs": ["generator: respuesta generada"],
    }


# ── Grafo ─────────────────────────────────────────────────────────────────────

def construir_grafo(con_scores: bool = False):
    builder = StateGraph(EstadoRetriever)

    retriever_fn = nodo_retriever_con_scores if con_scores else nodo_retriever_basico

    builder.add_node("retriever", retriever_fn)
    builder.add_node("generator", nodo_generator)

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 8.3: Nodo Retriever")

    console.print(Panel(
        "[bold]Responsabilidades del nodo retriever:[/]\n\n"
        "  1. Leer [cyan]pregunta[/] del estado\n"
        "  2. Ejecutar búsqueda en el vectorstore\n"
        "  3. Escribir [cyan]documentos[/] y [cyan]fuentes[/] en el estado\n"
        "  4. Registrar métricas (latencia, fuentes, scores)\n\n"
        "[dim]El nodo Grader (8.4) leerá 'documentos' para evaluar relevancia.\n"
        "Los scores de similitud son útiles para ese filtro.[/]",
        border_style="blue",
        title="Nodo Retriever",
    ))

    pregunta = "¿De qué trata el documento?"

    # Demo 1: retriever básico
    console.rule("[yellow]Retriever básico")
    grafo = construir_grafo(con_scores=False)
    estado: EstadoRetriever = {
        "pregunta": pregunta,
        "documentos": [],
        "fuentes": [],
        "contexto": "",
        "respuesta": "",
        "latencia_retriever_ms": 0.0,
        "logs": [],
    }
    resultado = grafo.invoke(estado)
    console.print(f"  Latencia: {resultado['latencia_retriever_ms']:.0f}ms")
    console.print(f"  Fuentes: {resultado['fuentes']}")

    # Demo 2: retriever con scores
    console.rule("[yellow]Retriever con scores de similitud")
    grafo_scores = construir_grafo(con_scores=True)
    estado2 = estado.copy()
    resultado2 = grafo_scores.invoke(estado2)

    console.print(Panel(
        resultado2["respuesta"][:300] + "...",
        title="Respuesta final",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
