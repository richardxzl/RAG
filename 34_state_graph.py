"""
34_state_graph.py — Módulo 7.1: StateGraph, estado, nodos y edges

LangGraph es un framework para construir aplicaciones LLM con estado, ciclos
y múltiples actores. La unidad central es el StateGraph:

  - State:  un TypedDict que define qué datos viven en el grafo
  - Nodes:  funciones que reciben el estado y retornan un dict parcial
  - Edges:  conexiones entre nodos (directas o condicionales)

Este script construye el grafo más simple posible:
  START → nodo_a → nodo_b → END
"""
from typing import TypedDict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langgraph.graph import StateGraph, START, END

console = Console()


# ── 1. Definir el Estado ──────────────────────────────────────────────────────
# El estado es la "memoria compartida" entre todos los nodos.
# Cada nodo lee del estado y devuelve un dict con las keys que quiere actualizar.

class MiEstado(TypedDict):
    mensaje: str
    pasos: list[str]
    contador: int


# ── 2. Definir los Nodos ──────────────────────────────────────────────────────
# Un nodo es cualquier función que:
#   - Recibe el estado completo como argumento
#   - Retorna un dict con las keys a actualizar (puede ser parcial)

def nodo_a(estado: MiEstado) -> dict:
    """Primer nodo: modifica el mensaje y registra que pasó por aquí."""
    console.print("  [cyan]→ ejecutando nodo_a[/]")
    return {
        "mensaje": estado["mensaje"].upper(),
        "pasos": estado["pasos"] + ["nodo_a"],
        "contador": estado["contador"] + 1,
    }


def nodo_b(estado: MiEstado) -> dict:
    """Segundo nodo: agrega un prefijo y registra el paso."""
    console.print("  [green]→ ejecutando nodo_b[/]")
    return {
        "mensaje": f"[procesado] {estado['mensaje']}",
        "pasos": estado["pasos"] + ["nodo_b"],
        "contador": estado["contador"] + 1,
    }


# ── 3. Construir el Grafo ─────────────────────────────────────────────────────

def construir_grafo() -> StateGraph:
    builder = StateGraph(MiEstado)

    # Registrar nodos
    builder.add_node("nodo_a", nodo_a)
    builder.add_node("nodo_b", nodo_b)

    # Conectar: START → nodo_a → nodo_b → END
    builder.add_edge(START, "nodo_a")
    builder.add_edge("nodo_a", "nodo_b")
    builder.add_edge("nodo_b", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]RAG Lab — Módulo 7.1: StateGraph")

    console.print(Panel(
        "[bold]Conceptos clave:[/]\n\n"
        "  [cyan]State[/]  → TypedDict compartido entre todos los nodos\n"
        "  [cyan]Node[/]   → función que lee el estado y retorna un dict parcial\n"
        "  [cyan]Edge[/]   → conexión entre nodos (START, END, o nombre de nodo)\n\n"
        "[dim]LangGraph hace un 'merge' del dict que retorna cada nodo\n"
        "con el estado actual. No necesitas retornar todo el estado.[/]",
        border_style="blue",
        title="StateGraph",
    ))

    grafo = construir_grafo()

    # Estado inicial
    estado_inicial: MiEstado = {
        "mensaje": "hola mundo",
        "pasos": [],
        "contador": 0,
    }

    console.print(f"\n[bold]Estado inicial:[/] {estado_inicial}")
    console.print("\n[bold]Ejecutando grafo...[/]")

    estado_final = grafo.invoke(estado_inicial)

    console.print(f"\n[bold]Estado final:[/]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Key")
    table.add_column("Valor")

    for key, val in estado_final.items():
        table.add_row(key, str(val))

    console.print(table)

    # Mostrar el grafo en texto
    console.print(Panel(
        grafo.get_graph().draw_ascii(),
        title="Grafo (ASCII)",
        border_style="yellow",
    ))

    # Tabla comparativa LCEL vs LangGraph
    tabla_comp = Table(title="LCEL vs LangGraph", show_header=True, header_style="bold")
    tabla_comp.add_column("Aspecto", style="bold")
    tabla_comp.add_column("LCEL")
    tabla_comp.add_column("LangGraph")

    tabla_comp.add_row("Flujo", "Lineal (pipe |)", "Grafo (nodos + edges)")
    tabla_comp.add_row("Estado", "Dict pasado manualmente", "TypedDict centralizado")
    tabla_comp.add_row("Ciclos", "No soporta", "Soporta (loops)")
    tabla_comp.add_row("Condicionales", "RunnableBranch", "Conditional edges")
    tabla_comp.add_row("Persistencia", "Manual", "Checkpointing integrado")
    tabla_comp.add_row("Multi-agente", "No", "Sí (subgraphs)")

    console.print(tabla_comp)


if __name__ == "__main__":
    main()
