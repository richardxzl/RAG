"""
49_tools_custom.py — Módulo 9.4: Crear tools custom — RAG, web, cálculos

Las herramientas son la interfaz entre el agente y el mundo exterior.
Un agente sin herramientas es simplemente un LLM con contexto.

Herramientas implementadas en este módulo:
  1. RAG tool:     busca en los documentos del proyecto
  2. Calculadora:  evaluación de expresiones matemáticas
  3. Web scraper:  simula búsqueda web (real con requests/httpx)
  4. Tool con estado: herramienta que mantiene historial
  5. Tool async:   para pipelines asíncronos

Anatomy de una herramienta:
  @tool
  def nombre(argumento: tipo) -> str:
      '''Descripción clara — el LLM la lee para decidir cuándo usar esta tool.'''
      ...
      return str(resultado)
"""
import math
import time
import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from rag.chain import get_llm, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Tool 1: RAG — buscar en documentos ────────────────────────────────────────

@tool
def buscar_en_documentos(pregunta: str) -> str:
    """
    Busca información relevante en la base de conocimiento del proyecto.
    Úsala cuando el usuario pregunta sobre el contenido de documentos específicos
    o necesita información que puede estar en los archivos cargados.
    """
    retriever = get_retriever()
    docs = retriever.invoke(pregunta)
    if not docs:
        return "No encontré documentos relevantes para esa pregunta."
    contexto = format_docs(docs[:3])  # limitar a 3 docs para no saturar el contexto
    return f"Información encontrada:\n{contexto[:1500]}"


# ── Tool 2: Calculadora avanzada ──────────────────────────────────────────────

@tool
def calcular(expresion: str) -> str:
    """
    Evalúa expresiones matemáticas. Soporta: +, -, *, /, **, sqrt, log, sin, cos, pi.
    Ejemplos: '2 ** 10', 'sqrt(144)', 'log(100, 10)', 'pi * 5 ** 2'.
    """
    funciones_seguras = {
        "sqrt": math.sqrt,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        resultado = eval(expresion, {"__builtins__": {}}, funciones_seguras)
        return f"{expresion} = {resultado}"
    except Exception as e:
        return f"Error al evaluar '{expresion}': {e}"


# ── Tool 3: Web search simulada ───────────────────────────────────────────────

@tool
def buscar_en_web(consulta: str) -> str:
    """
    Busca información actualizada en internet sobre un tema.
    Úsala para información reciente que el LLM podría no tener en su conocimiento.
    """
    # En producción: reemplazar con requests a una API real (SerpAPI, Tavily, etc.)
    resultados_simulados = {
        "langgraph": "LangGraph 1.1.3 es la versión estable. Soporta checkpointing, subgraphs y human-in-the-loop.",
        "python": "Python 3.13 fue lanzado en octubre 2024. Incluye mejoras en el GIL y nuevo compilador JIT.",
        "openai": "OpenAI lanzó GPT-4o mini en julio 2024. Claude Sonnet 4.6 es el modelo Anthropic actual.",
        "langchain": "LangChain 0.3 usa LCEL como interfaz principal. LangGraph es el componente de grafos.",
    }
    consulta_lower = consulta.lower()
    for key, val in resultados_simulados.items():
        if key in consulta_lower:
            return f"Web: {val}"
    return f"Búsqueda web para '{consulta}': no encontré resultados específicos en mi simulación."


# ── Tool 4: Tool con estado (usando closure) ──────────────────────────────────

def crear_tool_con_historial():
    """
    Las tools stateless son más simples, pero a veces necesitas estado.
    La forma idiomática es usar un closure o una clase.
    """
    historial: list[str] = []

    @tool
    def recordar(dato: str) -> str:
        """Guarda un dato para recordar durante la conversación."""
        historial.append(dato)
        return f"Guardado: '{dato}'. Total recordados: {len(historial)}"

    @tool
    def listar_recordados() -> str:
        """Lista todos los datos que se han pedido recordar."""
        if not historial:
            return "No hay nada recordado aún."
        return "Recordados:\n" + "\n".join(f"  - {d}" for d in historial)

    return recordar, listar_recordados


# ── Tool 5: Tool con validación de args ───────────────────────────────────────

@tool
def convertir_unidades(valor: float, de: str, a: str) -> str:
    """
    Convierte unidades de medida.
    Longitud: km, m, cm, mm, mi, ft, in
    Temperatura: celsius, fahrenheit, kelvin
    """
    # Longitud (todo a metros primero)
    longitud_a_m = {
        "km": 1000, "m": 1, "cm": 0.01, "mm": 0.001,
        "mi": 1609.34, "ft": 0.3048, "in": 0.0254,
    }
    de_lower, a_lower = de.lower(), a.lower()

    if de_lower in longitud_a_m and a_lower in longitud_a_m:
        en_metros = valor * longitud_a_m[de_lower]
        resultado = en_metros / longitud_a_m[a_lower]
        return f"{valor} {de} = {resultado:.4f} {a}"

    # Temperatura
    if de_lower == "celsius" and a_lower == "fahrenheit":
        return f"{valor}°C = {valor * 9/5 + 32:.2f}°F"
    if de_lower == "fahrenheit" and a_lower == "celsius":
        return f"{valor}°F = {(valor - 32) * 5/9:.2f}°C"
    if de_lower == "celsius" and a_lower == "kelvin":
        return f"{valor}°C = {valor + 273.15:.2f}K"

    return f"No sé convertir de '{de}' a '{a}'"


# ── Agente con todas las tools ────────────────────────────────────────────────

def construir_agente_completo():
    recordar, listar_recordados = crear_tool_con_historial()
    tools = [
        buscar_en_documentos,
        calcular,
        buscar_en_web,
        convertir_unidades,
        recordar,
        listar_recordados,
    ]
    llm = get_llm()
    return create_react_agent(llm, tools), tools


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 9.4: Tools Custom")

    console.print(Panel(
        "[bold]Anatomy de una tool:[/]\n\n"
        "[dim]@tool\n"
        "def nombre(argumento: tipo) -> str:\n"
        "    '''Descripción — el LLM la lee para decidir cuándo usar esta tool.'''\n"
        "    resultado = hacer_algo(argumento)\n"
        "    return str(resultado)  # siempre retorna string[/]\n\n"
        "[bold]Reglas:[/]\n"
        "  • El docstring ES la descripción que ve el LLM\n"
        "  • Los type hints se convierten en JSON schema\n"
        "  • Siempre retorna string (o serializable a string)\n"
        "  • El nombre debe ser descriptivo (es parte del schema)",
        border_style="blue",
        title="Tool custom",
    ))

    agente, tools = construir_agente_completo()

    # Mostrar tools disponibles
    tabla = Table(title="Tools del agente", show_header=True, header_style="bold magenta")
    tabla.add_column("Tool")
    tabla.add_column("Descripción")
    for t in tools:
        tabla.add_row(t.name, t.description[:70])
    console.print(tabla)

    # Demo de preguntas que usan distintas tools
    preguntas = [
        "¿Cuánto es la raíz cuadrada de 2025 multiplicada por pi?",
        "¿De qué trata el documento cargado en la base de conocimiento?",
        "¿Cuántos pies son 100 metros? ¿Y 37 grados Celsius en Fahrenheit?",
        "Recuerda que mi nombre es Ricardo. Ahora dime qué has recordado.",
        "Busca información sobre LangGraph en la web",
    ]

    for pregunta in preguntas:
        console.rule(f"[bold]{pregunta[:60]}")

        resultado = agente.invoke({
            "messages": [HumanMessage(content=pregunta)]
        })

        # Mostrar qué tools se usaron
        tools_usadas = []
        for msg in resultado["messages"]:
            if isinstance(msg, HumanMessage):
                continue
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_usadas.append(tc["name"])

        console.print(f"  [dim]Tools usadas: {tools_usadas or ['ninguna']}[/]")
        console.print(Panel(
            resultado["messages"][-1].content[:300],
            border_style="green",
        ))


if __name__ == "__main__":
    main()
