"""
35_nodes.py — Módulo 7.2: Nodes — funciones que transforman el estado

Un nodo en LangGraph es una función Python que:
  1. Recibe el estado completo (TypedDict)
  2. Hace algo (llama al LLM, ejecuta lógica, transforma datos)
  3. Retorna un dict con SOLO las keys que quiere actualizar

El grafo hace un "merge" shallow: las keys que retornas se mezclan
con el estado existente. Las que no retornas no cambian.

Patrones cubiertos:
  - Nodo simple (transformación pura)
  - Nodo con LLM
  - Nodo con efecto secundario (logging)
  - Nodo que lee múltiples keys del estado
  - Nodo que acumula en una lista (reducer pattern)
"""
import logging
from typing import TypedDict, Annotated
import operator

from rich.console import Console
from rich.panel import Panel

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm

console = Console()


# ── Estado con distintos tipos de datos ───────────────────────────────────────

class EstadoCompleto(TypedDict):
    pregunta: str
    respuesta: str
    intentos: int
    # Annotated + operator.add = acumulación automática (no reemplaza, suma)
    logs: Annotated[list[str], operator.add]


# ── Nodo 1: transformación pura ───────────────────────────────────────────────

def normalizar_pregunta(estado: EstadoCompleto) -> dict:
    """
    Nodo sin LLM: limpia y normaliza la pregunta.
    No necesita saber nada más del estado — solo lee 'pregunta'.
    """
    pregunta = estado["pregunta"].strip().lower()
    if not pregunta.endswith("?"):
        pregunta += "?"

    return {
        "pregunta": pregunta,
        "logs": [f"normalizar_pregunta: '{estado['pregunta']}' → '{pregunta}'"],
    }


# ── Nodo 2: llamada al LLM ────────────────────────────────────────────────────

def generar_respuesta(estado: EstadoCompleto) -> dict:
    """
    Nodo con LLM: genera una respuesta a la pregunta normalizada.
    Lee 'pregunta' del estado, escribe 'respuesta' e incrementa 'intentos'.
    """
    llm = get_llm()
    respuesta = llm.invoke([HumanMessage(content=estado["pregunta"])]).content

    return {
        "respuesta": respuesta,
        "intentos": estado["intentos"] + 1,
        "logs": [f"generar_respuesta: {len(respuesta)} caracteres generados"],
    }


# ── Nodo 3: efecto secundario + pass-through ──────────────────────────────────

def auditar(estado: EstadoCompleto) -> dict:
    """
    Nodo de auditoría: registra el resultado pero no modifica datos clave.
    Solo agrega al log. Útil para observabilidad sin acoplar al flujo.
    """
    log_entry = (
        f"AUDIT: pregunta='{estado['pregunta'][:40]}...', "
        f"intentos={estado['intentos']}, "
        f"respuesta_len={len(estado.get('respuesta', ''))}"
    )
    console.print(f"  [dim]{log_entry}[/]")
    return {
        "logs": [log_entry],
    }


# ── Grafo ─────────────────────────────────────────────────────────────────────

def construir_grafo():
    builder = StateGraph(EstadoCompleto)

    builder.add_node("normalizar", normalizar_pregunta)
    builder.add_node("generar", generar_respuesta)
    builder.add_node("auditar", auditar)

    builder.add_edge(START, "normalizar")
    builder.add_edge("normalizar", "generar")
    builder.add_edge("generar", "auditar")
    builder.add_edge("auditar", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 7.2: Nodes")

    console.print(Panel(
        "[bold]Reglas de los nodos:[/]\n\n"
        "  1. Reciben [cyan]todo el estado[/] como argumento\n"
        "  2. Retornan [cyan]solo las keys que cambian[/] (merge parcial)\n"
        "  3. Pueden ser [cyan]funciones puras[/] o llamar al LLM\n"
        "  4. [cyan]Annotated + operator.add[/] → acumulación automática en listas\n\n"
        "[dim]Un nodo NO necesita retornar el estado completo.\n"
        "Si retorna {'respuesta': 'x'}, solo 'respuesta' cambia.[/]",
        border_style="blue",
        title="Nodos en LangGraph",
    ))

    grafo = construir_grafo()

    estado_inicial: EstadoCompleto = {
        "pregunta": "  qué es LangGraph  ",
        "respuesta": "",
        "intentos": 0,
        "logs": [],
    }

    console.print(f"\n[bold]Input:[/] '{estado_inicial['pregunta']}'")
    console.print("\n[bold]Ejecutando nodos...[/]")

    resultado = grafo.invoke(estado_inicial)

    console.print(Panel(
        resultado["respuesta"][:400] + "...",
        title="Respuesta del LLM",
        border_style="green",
    ))

    console.print(Panel(
        "\n".join(f"  • {log}" for log in resultado["logs"]),
        title=f"Logs acumulados ({len(resultado['logs'])} entradas)",
        border_style="yellow",
    ))

    console.print(f"\n[dim]Intentos: {resultado['intentos']}[/]")

    # Patrón Annotated explicado
    console.print(Panel(
        "[bold]Annotated[operator.add] — reducer pattern:[/]\n\n"
        "  Sin Annotated:  cada nodo REEMPLAZA la lista\n"
        "  Con Annotated:  cada nodo ACUMULA en la lista\n\n"
        "  [dim]logs: Annotated[list[str], operator.add]\n"
        "  → nodo_a retorna {'logs': ['a']}\n"
        "  → nodo_b retorna {'logs': ['b']}\n"
        "  → estado final: logs = ['a', 'b'] (no solo ['b'])[/]",
        border_style="magenta",
    ))


if __name__ == "__main__":
    main()
