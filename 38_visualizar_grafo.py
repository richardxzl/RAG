"""
38_visualizar_grafo.py — Módulo 7.5: Visualizar el grafo (Mermaid diagram)

LangGraph puede exportar la estructura del grafo como:
  - Mermaid (texto): para pegar en Obsidian, GitHub, etc.
  - ASCII art: para ver en la terminal
  - PNG (requiere graphviz instalado)

La visualización sirve para:
  1. Documentar la arquitectura del sistema
  2. Debuggear ciclos o edges incorrectos
  3. Comunicar el flujo a stakeholders no técnicos

Este script construye un grafo más complejo (con condicionales)
para que la visualización sea interesante.
"""
from typing import TypedDict, Literal

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from langgraph.graph import StateGraph, START, END

console = Console()


# ── Grafo de ejemplo: RAG simplificado con router ─────────────────────────────

class EstadoRAG(TypedDict):
    pregunta: str
    necesita_rag: bool
    contexto: str
    respuesta: str


def nodo_router(estado: EstadoRAG) -> dict:
    """Decide si la pregunta necesita buscar en documentos."""
    preguntas_directas = ["hola", "gracias", "ayuda", "cómo estás"]
    necesita = not any(p in estado["pregunta"].lower() for p in preguntas_directas)
    return {"necesita_rag": necesita}


def nodo_retriever(estado: EstadoRAG) -> dict:
    """Busca documentos relevantes (simulado)."""
    return {"contexto": f"[Contexto recuperado para: {estado['pregunta'][:30]}...]"}


def nodo_generator(estado: EstadoRAG) -> dict:
    """Genera respuesta con contexto."""
    return {"respuesta": f"[Con RAG] Respuesta basada en: {estado['contexto'][:40]}..."}


def nodo_directo(estado: EstadoRAG) -> dict:
    """Responde directamente sin RAG."""
    return {"respuesta": f"[Directo] Respuesta conversacional a: {estado['pregunta']}"}


def decidir_ruta(estado: EstadoRAG) -> Literal["retriever", "directo"]:
    return "retriever" if estado["necesita_rag"] else "directo"


def construir_grafo_rag():
    builder = StateGraph(EstadoRAG)

    builder.add_node("router", nodo_router)
    builder.add_node("retriever", nodo_retriever)
    builder.add_node("generator", nodo_generator)
    builder.add_node("directo", nodo_directo)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", decidir_ruta, {
        "retriever": "retriever",
        "directo": "directo",
    })
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)
    builder.add_edge("directo", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]RAG Lab — Módulo 7.5: Visualizar el grafo")

    grafo = construir_grafo_rag()
    graph_repr = grafo.get_graph()

    # ── ASCII (funciona siempre en terminal) ──────────────────────────────────
    console.print(Panel(
        graph_repr.draw_ascii(),
        title="Visualización ASCII",
        border_style="yellow",
    ))

    # ── Mermaid ───────────────────────────────────────────────────────────────
    mermaid_str = graph_repr.draw_mermaid()
    console.print(Panel(
        Syntax(mermaid_str, "text", theme="monokai"),
        title="Mermaid diagram (pegar en Obsidian / GitHub)",
        border_style="blue",
    ))

    # ── Cómo insertar en Obsidian ─────────────────────────────────────────────
    obsidian_snippet = f"```mermaid\n{mermaid_str}\n```"
    console.print(Panel(
        Syntax(obsidian_snippet, "markdown", theme="monokai"),
        title="Snippet para Obsidian / GitHub Markdown",
        border_style="green",
    ))

    # ── PNG (opcional, requiere graphviz) ─────────────────────────────────────
    console.print(Panel(
        "[bold]Guardar como PNG (requiere: pip install pygraphviz):[/]\n\n"
        "[dim]from IPython.display import Image\n"
        "png_data = grafo.get_graph().draw_mermaid_png()\n"
        "with open('grafo.png', 'wb') as f:\n"
        "    f.write(png_data)[/]\n\n"
        "[yellow]O con graphviz directamente:[/]\n"
        "[dim]grafo.get_graph(xray=True).draw_png('grafo_detallado.png')[/]",
        border_style="magenta",
        title="Exportar PNG",
    ))

    # ── Introspección de nodos y edges ────────────────────────────────────────
    console.print(Panel(
        f"[bold]Nodos:[/] {[n.id for n in graph_repr.nodes.values()]}\n\n"
        f"[bold]Edges:[/]\n" +
        "\n".join(
            f"  {e.source} → {e.target}"
            + (f" [dim](condicional)[/]" if e.conditional else "")
            for e in graph_repr.edges
        ),
        title="Introspección del grafo",
        border_style="cyan",
    ))


if __name__ == "__main__":
    main()
