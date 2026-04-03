"""
48_react_pattern.py — Módulo 9.3: ReAct pattern — Reasoning + Acting en loop

ReAct (Reasoning + Acting) es el patrón de agente más usado en LLMs.
El LLM alterna entre:
  - Thought:  razonamiento sobre qué hacer
  - Action:   llamar a una herramienta
  - Observation: resultado de la herramienta
  → repite hasta tener suficiente información para responder

LangGraph implementa esto con un grafo de dos nodos en loop:
  llm_node ←→ tools_node

Este script implementa el patrón manualmente para entenderlo,
y luego muestra create_react_agent como alternativa compacta.
"""
import logging
from typing import TypedDict, Annotated

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent

from rag.chain import get_llm

console = Console()


# ── Herramientas ──────────────────────────────────────────────────────────────

@tool
def buscar_poblacion(ciudad: str) -> str:
    """Busca la población aproximada de una ciudad."""
    datos = {
        "ciudad de mexico": "Ciudad de México: ~22 millones (zona metropolitana)",
        "buenos aires": "Buenos Aires: ~15 millones (zona metropolitana)",
        "madrid": "Madrid: ~6.7 millones (zona metropolitana)",
        "barcelona": "Barcelona: ~5.6 millones (zona metropolitana)",
        "bogota": "Bogotá: ~11 millones",
    }
    return datos.get(ciudad.lower(), f"No tengo datos de población para '{ciudad}'")


@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática. Ejemplo: '22000000 / 15000000'."""
    try:
        resultado = eval(expresion, {"__builtins__": {}}, {})
        return f"{expresion} = {resultado:.4f}"
    except Exception as e:
        return f"Error: {e}"


@tool
def buscar_capital(pais: str) -> str:
    """Retorna la capital de un país."""
    capitales = {
        "mexico": "Ciudad de México",
        "argentina": "Buenos Aires",
        "españa": "Madrid",
        "colombia": "Bogotá",
        "peru": "Lima",
        "chile": "Santiago",
    }
    return capitales.get(pais.lower(), f"No tengo datos para '{pais}'")


TOOLS = [buscar_poblacion, calcular, buscar_capital]


# ── Implementación manual del ReAct ──────────────────────────────────────────
# LangGraph representa el estado del agente como una lista de mensajes.
# El loop: llm_node → tools_node → llm_node → ... hasta que el LLM no pide tools.

class EstadoReAct(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]


def nodo_llm(estado: EstadoReAct) -> dict:
    """El LLM razona y decide si llamar una herramienta o responder."""
    llm = get_llm()
    llm_con_tools = llm.bind_tools(TOOLS)
    respuesta = llm_con_tools.invoke(estado["messages"])

    # Logging del reasoning
    if respuesta.tool_calls:
        for tc in respuesta.tool_calls:
            console.print(f"  [yellow]→ Acción:[/] {tc['name']}({tc['args']})")
    else:
        console.print(f"  [green]→ Respuesta final[/]")

    return {"messages": [respuesta]}


def necesita_tools(estado: EstadoReAct) -> str:
    """¿El último mensaje del LLM pide herramientas?"""
    ultimo = estado["messages"][-1]
    if isinstance(ultimo, AIMessage) and ultimo.tool_calls:
        return "tools"
    return END


def construir_react_manual():
    """
    Grafo ReAct construido a mano.
    El loop: llm → tools → llm → tools → ... → END
    """
    tool_node = ToolNode(TOOLS)

    builder = StateGraph(EstadoReAct)
    builder.add_node("llm", nodo_llm)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", necesita_tools, {
        "tools": "tools",
        END: END,
    })
    builder.add_edge("tools", "llm")  # ← el loop

    return builder.compile()


# ── create_react_agent (versión compacta) ─────────────────────────────────────

def construir_react_prebuilt():
    """
    create_react_agent hace exactamente lo mismo que construir_react_manual()
    pero en una sola línea. Útil para prototipado rápido.
    """
    llm = get_llm()
    return create_react_agent(llm, TOOLS)


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 9.3: ReAct Pattern")

    console.print(Panel(
        "[bold]ReAct = Reasoning + Acting[/]\n\n"
        "El LLM alterna entre:\n"
        "  [yellow]Thought[/]:      'Necesito la población de ambas ciudades'\n"
        "  [yellow]Action[/]:       buscar_poblacion('ciudad de mexico')\n"
        "  [yellow]Observation[/]:  '~22 millones'\n"
        "  [yellow]Action[/]:       buscar_poblacion('buenos aires')\n"
        "  [yellow]Observation[/]:  '~15 millones'\n"
        "  [yellow]Action[/]:       calcular('22000000 / 15000000')\n"
        "  [yellow]Observation[/]:  '1.4667'\n"
        "  [green]Answer[/]:        'Ciudad de México tiene ~1.47x la población de Buenos Aires'\n\n"
        "[dim]El loop continúa hasta que el LLM tiene suficiente información.[/]",
        border_style="blue",
        title="ReAct Loop",
    ))

    # Mostrar el grafo
    agente_manual = construir_react_manual()
    console.print(Panel(
        agente_manual.get_graph().draw_ascii(),
        title="Grafo ReAct (manual)",
        border_style="yellow",
    ))

    preguntas = [
        "¿Cuántas veces más grande en población es Ciudad de México vs Madrid?",
        "¿Cuál es la capital de Argentina y cuánta población tiene?",
        "Compara la población de Bogotá y Barcelona",
    ]

    for pregunta in preguntas:
        console.rule(f"[bold]{pregunta}")
        console.print("[dim]Pasos del agente:[/]")

        resultado = agente_manual.invoke({
            "messages": [HumanMessage(content=pregunta)]
        })

        respuesta_final = resultado["messages"][-1].content
        pasos = sum(
            1 for m in resultado["messages"]
            if isinstance(m, AIMessage) and m.tool_calls
        )

        console.print(Panel(
            f"[dim]Pasos de razonamiento: {pasos}[/]\n\n{respuesta_final}",
            title="Respuesta final",
            border_style="green",
        ))

    # Comparar manual vs prebuilt
    console.rule("[yellow]create_react_agent vs implementación manual")
    console.print(Panel(
        "[bold]construir_react_manual()[/]:\n"
        "  StateGraph + nodo_llm + ToolNode + conditional_edges\n"
        "  ~30 líneas de código\n\n"
        "[bold]create_react_agent(llm, tools)[/]:\n"
        "  Una línea. Hace exactamente lo mismo.\n\n"
        "[dim]Usa create_react_agent para prototipado.\n"
        "Construye el grafo manual cuando necesites personalizar\n"
        "(añadir nodos extra, cambiar la lógica de routing, etc.)[/]",
        border_style="cyan",
    ))

    # Verificar que ambos producen el mismo grafo
    agente_prebuilt = construir_react_prebuilt()
    console.print(Panel(
        agente_prebuilt.get_graph().draw_ascii(),
        title="Grafo create_react_agent (prebuilt)",
        border_style="dim",
    ))


if __name__ == "__main__":
    main()
