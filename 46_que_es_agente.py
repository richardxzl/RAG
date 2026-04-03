"""
46_que_es_agente.py — Módulo 9.1: ¿Qué es un agente?

Una CHAIN ejecuta siempre los mismos pasos en el mismo orden.
Un AGENTE decide EN TIEMPO DE EJECUCIÓN qué hacer, cuándo hacerlo y si repetirlo.

La diferencia clave: el LLM controla el flujo de ejecución.

  Chain:  input → paso1 → paso2 → paso3 → output  (hardcoded)
  Agente: input → LLM decide → [herramienta A | herramienta B | responder] → loop

Este script demuestra la diferencia conceptual construyendo ambos y
comparando cómo responden ante preguntas que requieren razonamiento.
"""
import logging
from typing import TypedDict, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from rag.chain import get_llm

console = Console()


# ── Herramientas disponibles para el agente ───────────────────────────────────

@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática simple. Ejemplo: '2 + 2', '10 * 5'."""
    try:
        resultado = eval(expresion, {"__builtins__": {}}, {})
        return f"{expresion} = {resultado}"
    except Exception as e:
        return f"Error al calcular '{expresion}': {e}"


@tool
def buscar_definicion(termino: str) -> str:
    """Busca la definición de un término técnico de IA/ML."""
    definiciones = {
        "agente": "Sistema de IA que percibe el entorno y toma acciones para alcanzar un objetivo.",
        "llm": "Large Language Model: modelo de lenguaje entrenado sobre grandes volúmenes de texto.",
        "rag": "Retrieval-Augmented Generation: técnica que combina búsqueda de información con generación de texto.",
        "embedding": "Representación vectorial de texto en un espacio de alta dimensión.",
        "langgraph": "Framework para construir aplicaciones LLM con estado, ciclos y múltiples actores.",
        "react": "Reasoning + Acting: patrón donde el LLM razona antes de actuar en un loop.",
    }
    termino_lower = termino.lower()
    for key, val in definiciones.items():
        if key in termino_lower:
            return f"{termino}: {val}"
    return f"No encontré definición para '{termino}'. Términos disponibles: {list(definiciones.keys())}"


@tool
def contar_palabras(texto: str) -> str:
    """Cuenta el número de palabras en un texto."""
    palabras = len(texto.split())
    return f"El texto tiene {palabras} palabras."


# ── Demo: comparar Chain vs Agente ───────────────────────────────────────────

def demo_chain_vs_agente():
    """
    La chain SIEMPRE ejecuta retriever → generator, sin importar la pregunta.
    El agente DECIDE qué herramienta usar (o si responder directamente).
    """
    llm = get_llm()
    tools = [calcular, buscar_definicion, contar_palabras]

    # Agente ReAct (la forma más simple con LangGraph)
    agente = create_react_agent(llm, tools)

    preguntas = [
        "¿Cuánto es 347 multiplicado por 28?",
        "¿Qué es RAG?",
        "Cuenta las palabras de: el agente decide qué herramienta usar",
        "¿Cuánto es 15 elevado al cuadrado y qué significa LLM?",
        "Hola, ¿cómo estás?",  # sin herramientas necesarias
    ]

    tabla = Table(
        title="Agente: decisiones de herramientas",
        show_header=True,
        header_style="bold magenta",
    )
    tabla.add_column("Pregunta", max_width=40)
    tabla.add_column("Herramientas usadas")
    tabla.add_column("Respuesta (60 chars)")

    for pregunta in preguntas:
        resultado = agente.invoke({
            "messages": [HumanMessage(content=pregunta)]
        })

        # Extraer qué herramientas se usaron
        herramientas_usadas = []
        for msg in resultado["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    herramientas_usadas.append(tc["name"])

        respuesta_final = resultado["messages"][-1].content

        tabla.add_row(
            pregunta[:39],
            ", ".join(herramientas_usadas) or "ninguna (respuesta directa)",
            respuesta_final[:60].replace("\n", " ") + "...",
        )

    console.print(tabla)


# ── Mostrar el grafo interno del agente ──────────────────────────────────────

def mostrar_grafo_agente():
    llm = get_llm()
    tools = [calcular, buscar_definicion]
    agente = create_react_agent(llm, tools)

    console.print(Panel(
        agente.get_graph().draw_ascii(),
        title="Grafo interno del agente ReAct",
        border_style="yellow",
    ))


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 9.1: ¿Qué es un agente?")

    console.print(Panel(
        "[bold]Chain vs Agente:[/]\n\n"
        "  [cyan]Chain[/]:  pasos hardcoded. Siempre hace A → B → C.\n"
        "          No importa la pregunta — el flujo es fijo.\n\n"
        "  [cyan]Agente[/]: el LLM decide el flujo. En cada paso puede:\n"
        "          • Llamar a una herramienta (calcular, buscar, etc.)\n"
        "          • Procesar el resultado y decidir el siguiente paso\n"
        "          • Responder directamente sin herramientas\n"
        "          • Llamar múltiples herramientas en secuencia\n\n"
        "[bold]El componente clave:[/] tool calling\n"
        "El LLM genera un objeto estructurado con:\n"
        "  { name: 'calcular', args: { expresion: '347 * 28' } }\n"
        "El framework ejecuta la herramienta y devuelve el resultado.",
        border_style="blue",
        title="Chain vs Agente",
    ))

    mostrar_grafo_agente()

    console.print("\n[bold]Observa cómo el agente elige herramientas según la pregunta:[/]\n")
    demo_chain_vs_agente()

    console.print(Panel(
        "[bold]¿Cuándo usar agente vs chain?[/]\n\n"
        "  [cyan]Chain[/]:\n"
        "    • El flujo es siempre el mismo\n"
        "    • Necesitas máxima previsibilidad y control\n"
        "    • Latencia crítica (sin overhead de razonamiento)\n\n"
        "  [cyan]Agente[/]:\n"
        "    • El flujo depende de la pregunta o el contexto\n"
        "    • Tienes múltiples herramientas y el LLM sabe cuándo usar cada una\n"
        "    • Necesitas manejar preguntas que requieren múltiples pasos",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
