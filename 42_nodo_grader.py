"""
42_nodo_grader.py — Módulo 8.4: Nodo Grader — ¿los chunks son relevantes?

El Grader evalúa si los documentos recuperados son relevantes para la pregunta.
Si no lo son, en lugar de generar una respuesta de mala calidad, el sistema
puede reformular la pregunta y volver a buscar.

Este es el corazón del patrón "Corrective RAG":
  retriever → grader → [si relevante: generator] [si no: reformular → retriever]

Dos estrategias de grading:
  1. Score de similitud (umbral fijo, sin LLM)
  2. LLM judge (más preciso, evalúa semántica)
"""
from typing import TypedDict, Literal, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever, get_vectorstore
from rag.config import RETRIEVAL_K

console = Console()

# Umbral: score de similitud por encima del cual un doc se considera relevante
# ChromaDB usa distancia (menor = mejor), ~0.5 es un umbral razonable
SCORE_THRESHOLD = 0.7


# ── Estado ────────────────────────────────────────────────────────────────────

class EstadoGrader(TypedDict):
    pregunta: str
    documentos: list[Document]
    documentos_relevantes: list[Document]
    todos_relevantes: bool
    contexto: str
    respuesta: str
    logs: Annotated[list[str], operator.add]


# ── Nodo Retriever ────────────────────────────────────────────────────────────

def nodo_retriever(estado: EstadoGrader) -> dict:
    vs = get_vectorstore()
    resultados = vs.similarity_search_with_score(estado["pregunta"], k=RETRIEVAL_K)
    docs = []
    for doc, score in resultados:
        doc.metadata["retrieval_score"] = float(score)
        docs.append(doc)
    return {
        "documentos": docs,
        "logs": [f"retriever: {len(docs)} docs recuperados"],
    }


# ── Nodo Grader — estrategia 1: por score ────────────────────────────────────

def nodo_grader_score(estado: EstadoGrader) -> dict:
    """
    Filtra documentos por score de similitud.
    Rápido, sin LLM. Funciona bien si los embeddings son de buena calidad.
    """
    docs = estado["documentos"]
    relevantes = [
        doc for doc in docs
        if doc.metadata.get("retrieval_score", 1.0) <= SCORE_THRESHOLD
    ]

    tabla = Table(show_header=True, header_style="bold magenta", title="Grading por score")
    tabla.add_column("Score")
    tabla.add_column("Relevante")
    tabla.add_column("Preview")

    for doc in docs:
        score = doc.metadata.get("retrieval_score", "?")
        es_relevante = score <= SCORE_THRESHOLD if isinstance(score, float) else "?"
        tabla.add_row(
            f"{score:.4f}" if isinstance(score, float) else str(score),
            "[green]✓[/]" if es_relevante else "[red]✗[/]",
            doc.page_content[:50].replace("\n", " "),
        )
    console.print(tabla)

    todos = len(relevantes) == len(docs) and len(docs) > 0
    console.print(f"  [cyan]grader_score:[/] {len(relevantes)}/{len(docs)} relevantes")

    return {
        "documentos_relevantes": relevantes,
        "todos_relevantes": todos,
        "logs": [f"grader_score: {len(relevantes)}/{len(docs)} relevantes"],
    }


# ── Nodo Grader — estrategia 2: LLM judge ────────────────────────────────────

GRADER_PROMPT = ChatPromptTemplate.from_template(
    """Evalúa si el siguiente fragmento de documento es relevante para responder la pregunta.

Pregunta: {pregunta}

Fragmento:
{fragmento}

Responde SOLO con una palabra: "relevante" o "irrelevante"."""
)


def nodo_grader_llm(estado: EstadoGrader) -> dict:
    """
    Evalúa relevancia con el LLM.
    Más preciso que el score — entiende paráfrasis y contexto semántico.
    Más lento: una llamada al LLM por documento.
    """
    llm = get_llm()
    cadena = GRADER_PROMPT | llm
    relevantes = []

    tabla = Table(show_header=True, header_style="bold magenta", title="Grading con LLM")
    tabla.add_column("Veredicto")
    tabla.add_column("Preview")

    for doc in estado["documentos"]:
        resultado = cadena.invoke({
            "pregunta": estado["pregunta"],
            "fragmento": doc.page_content[:500],
        }).content.strip().lower()

        es_relevante = "relevante" in resultado and "irrelevante" not in resultado
        if es_relevante:
            relevantes.append(doc)

        tabla.add_row(
            "[green]relevante[/]" if es_relevante else "[red]irrelevante[/]",
            doc.page_content[:50].replace("\n", " "),
        )

    console.print(tabla)

    todos = len(relevantes) == len(estado["documentos"]) and len(estado["documentos"]) > 0

    return {
        "documentos_relevantes": relevantes,
        "todos_relevantes": todos,
        "logs": [f"grader_llm: {len(relevantes)}/{len(estado['documentos'])} relevantes"],
    }


# ── Nodo Generator ────────────────────────────────────────────────────────────

def nodo_generator(estado: EstadoGrader) -> dict:
    docs = estado["documentos_relevantes"] or estado["documentos"]
    contexto = format_docs(docs)
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


# ── Función de routing post-grader ────────────────────────────────────────────

def decidir_post_grader(estado: EstadoGrader) -> Literal["generator", END]:
    if estado["documentos_relevantes"]:
        return "generator"
    # Si no hay docs relevantes, generamos igual (el loop de reformulación
    # se implementa en el módulo 8.7)
    console.print("  [yellow]grader: sin docs relevantes, generando de todos modos[/]")
    return "generator"


# ── Grafo ─────────────────────────────────────────────────────────────────────

def construir_grafo(usar_llm: bool = False):
    builder = StateGraph(EstadoGrader)

    grader_fn = nodo_grader_llm if usar_llm else nodo_grader_score

    builder.add_node("retriever", nodo_retriever)
    builder.add_node("grader", grader_fn)
    builder.add_node("generator", nodo_generator)

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "grader")
    builder.add_conditional_edges("grader", decidir_post_grader, {
        "generator": "generator",
        END: END,
    })
    builder.add_edge("generator", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]RAG Lab — Módulo 8.4: Nodo Grader")

    console.print(Panel(
        "[bold]¿Por qué un Grader?[/]\n\n"
        "El retriever devuelve los top-k documentos por similitud.\n"
        "Pero 'similar' no siempre significa 'relevante para responder'.\n\n"
        "  Sin Grader: usas todos los chunks aunque sean ruido\n"
        "  Con Grader: filtras los irrelevantes antes de generar\n\n"
        "[bold]Estrategias:[/]\n"
        "  [yellow]score[/]:  umbral de similitud (rápido, sin LLM)\n"
        "  [yellow]LLM[/]:    evaluación semántica (preciso, +costo)",
        border_style="blue",
        title="Nodo Grader",
    ))

    pregunta = "¿De qué trata el documento?"

    estado_base: EstadoGrader = {
        "pregunta": pregunta,
        "documentos": [],
        "documentos_relevantes": [],
        "todos_relevantes": False,
        "contexto": "",
        "respuesta": "",
        "logs": [],
    }

    console.rule("[yellow]Grader por score de similitud")
    grafo = construir_grafo(usar_llm=False)
    resultado = grafo.invoke(estado_base.copy())

    console.print(Panel(
        f"Docs recuperados: {len(resultado['documentos'])}\n"
        f"Docs relevantes: {len(resultado['documentos_relevantes'])}\n"
        f"Todos relevantes: {resultado['todos_relevantes']}",
        title="Resultado del grading",
        border_style="cyan",
    ))

    console.print(Panel(
        resultado["respuesta"][:300] + "...",
        title="Respuesta final",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
