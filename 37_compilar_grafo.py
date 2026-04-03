"""
37_compilar_grafo.py — Módulo 7.4: Compilar y ejecutar un grafo mínimo

El proceso completo de LangGraph es:
  1. Definir el State (TypedDict)
  2. Definir los nodos (funciones)
  3. Construir el grafo (StateGraph)
  4. Compilar → CompiledGraph (esto valida la estructura)
  5. Ejecutar con .invoke(), .stream(), o .batch()

Este script cubre las distintas formas de ejecutar el grafo compilado
y cómo inspeccionar el estado en cada paso.
"""
import logging
from typing import TypedDict, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm

console = Console()


# ── Estado y nodos mínimos ────────────────────────────────────────────────────

class EstadoMinimo(TypedDict):
    entrada: str
    salida: str
    pasos: Annotated[list[str], operator.add]


def paso_1(estado: EstadoMinimo) -> dict:
    texto = estado["entrada"].strip()
    return {
        "entrada": texto,
        "pasos": ["paso_1: texto normalizado"],
    }


def paso_2_llm(estado: EstadoMinimo) -> dict:
    llm = get_llm()
    respuesta = llm.invoke([
        HumanMessage(content=f"Resume en una oración: {estado['entrada']}")
    ]).content
    return {
        "salida": respuesta,
        "pasos": ["paso_2: LLM invocado"],
    }


def construir_grafo():
    builder = StateGraph(EstadoMinimo)
    builder.add_node("normalizar", paso_1)
    builder.add_node("resumir", paso_2_llm)
    builder.add_edge(START, "normalizar")
    builder.add_edge("normalizar", "resumir")
    builder.add_edge("resumir", END)
    return builder.compile()


# ── Formas de ejecución ───────────────────────────────────────────────────────

def demo_invoke(grafo, entrada: str):
    """
    .invoke() → retorna el estado FINAL completo.
    Bloqueante. La forma más simple.
    """
    console.rule("[yellow]invoke() — estado final")
    resultado = grafo.invoke({"entrada": entrada, "salida": "", "pasos": []})
    console.print(f"  salida: [green]{resultado['salida']}[/]")
    console.print(f"  pasos:  {resultado['pasos']}")


def demo_stream(grafo, entrada: str):
    """
    .stream() → itera sobre los estados parciales después de CADA nodo.
    Útil para ver el progreso o hacer streaming al usuario.
    """
    console.rule("[yellow]stream() — estado por nodo")
    for evento in grafo.stream(
        {"entrada": entrada, "salida": "", "pasos": []},
        stream_mode="updates",  # 'updates' → solo las keys que cambiaron
    ):
        for nodo, cambios in evento.items():
            console.print(f"  [cyan]{nodo}[/] actualizó: {list(cambios.keys())}")


def demo_stream_values(grafo, entrada: str):
    """
    stream_mode='values' → emite el estado COMPLETO después de cada nodo.
    """
    console.rule("[yellow]stream(mode='values') — estado completo por nodo")
    for estado in grafo.stream(
        {"entrada": entrada, "salida": "", "pasos": []},
        stream_mode="values",
    ):
        console.print(f"  estado actual: salida='{estado.get('salida', '')[:50]}'")


def demo_get_state(grafo, entrada: str):
    """
    Con un checkpointer, puedes pausar y obtener el estado en cualquier momento.
    Sin checkpointer, usamos stream() para inspeccionar.
    """
    console.rule("[yellow]Inspección paso a paso")
    tabla = Table(show_header=True, header_style="bold magenta")
    tabla.add_column("Después de nodo")
    tabla.add_column("pasos")
    tabla.add_column("salida")

    for evento in grafo.stream(
        {"entrada": entrada, "salida": "", "pasos": []},
        stream_mode="values",
    ):
        tabla.add_row(
            str(evento.get("pasos", [])[-1] if evento.get("pasos") else "—"),
            str(evento.get("pasos", [])),
            evento.get("salida", "")[:40] or "—",
        )
    console.print(tabla)


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 7.4: Compilar y ejecutar")

    console.print(Panel(
        "[bold]Proceso completo:[/]\n\n"
        "  1. [cyan]StateGraph(Estado)[/]    → builder vacío\n"
        "  2. [cyan].add_node()[/]            → registrar nodos\n"
        "  3. [cyan].add_edge()[/]            → conectar nodos\n"
        "  4. [cyan].compile()[/]             → valida + retorna CompiledGraph\n"
        "  5. [cyan].invoke() / .stream()[/]  → ejecutar\n\n"
        "[bold]Modos de ejecución:[/]\n"
        "  [cyan]invoke()[/]                → estado final (bloqueante)\n"
        "  [cyan]stream(mode='updates')[/]  → keys cambiadas por nodo\n"
        "  [cyan]stream(mode='values')[/]   → estado completo por nodo\n"
        "  [cyan]batch()[/]                 → múltiples inputs en paralelo",
        border_style="blue",
        title="Compilar y ejecutar un grafo",
    ))

    grafo = construir_grafo()
    entrada = "LangGraph es un framework de Python para construir aplicaciones LLM con estado y ciclos."

    demo_invoke(grafo, entrada)
    demo_stream(grafo, entrada)
    demo_stream_values(grafo, entrada)
    demo_get_state(grafo, entrada)

    # Batch: múltiples inputs en paralelo
    console.rule("[yellow]batch() — múltiples inputs")
    entradas = [
        {"entrada": "Python es un lenguaje de programación.", "salida": "", "pasos": []},
        {"entrada": "LangChain facilita la construcción de aplicaciones con LLMs.", "salida": "", "pasos": []},
    ]
    resultados = grafo.batch(entradas)
    for i, r in enumerate(resultados):
        console.print(f"  [dim]Input {i+1}:[/] {r['salida'][:80]}")


if __name__ == "__main__":
    main()
