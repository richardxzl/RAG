"""
47_tool_calling.py — Módulo 9.2: Tool calling — el LLM decide qué herramientas usar

Tool calling (o function calling) es el mecanismo que permite a un LLM
generar llamadas a funciones estructuradas en vez de texto libre.

Flujo:
  1. Le pasas al LLM una lista de herramientas (nombre + descripción + schema)
  2. El LLM decide si necesita alguna y genera un AIMessage con tool_calls
  3. Tu código ejecuta la herramienta y genera un ToolMessage con el resultado
  4. El LLM procesa el resultado y genera la respuesta final

Este módulo explora tool calling a bajo nivel — sin create_react_agent —
para entender exactamente qué pasa en cada paso.
"""
import json
import logging
from typing import TypedDict, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm

console = Console()


# ── Definición de herramientas ────────────────────────────────────────────────
# @tool usa el docstring como descripción y los type hints como schema.
# El LLM lee la descripción para saber cuándo usar cada herramienta.

@tool
def obtener_clima(ciudad: str) -> str:
    """Obtiene el clima actual de una ciudad. Usa esto cuando el usuario pregunta por el tiempo."""
    climas = {
        "madrid": "☀️ Madrid: 22°C, soleado",
        "barcelona": "⛅ Barcelona: 18°C, parcialmente nublado",
        "buenos aires": "🌧️ Buenos Aires: 15°C, lluvia",
        "ciudad de mexico": "🌤️ Ciudad de México: 25°C, despejado",
    }
    return climas.get(ciudad.lower(), f"No tengo datos del clima para '{ciudad}'")


@tool
def convertir_moneda(monto: float, de: str, a: str) -> str:
    """
    Convierte un monto de una moneda a otra.
    Monedas soportadas: USD, EUR, MXN, ARS, COP.
    """
    tasas_a_usd = {"USD": 1.0, "EUR": 1.08, "MXN": 0.058, "ARS": 0.001, "COP": 0.00025}
    de_upper = de.upper()
    a_upper = a.upper()
    if de_upper not in tasas_a_usd or a_upper not in tasas_a_usd:
        return f"Moneda no soportada. Opciones: {list(tasas_a_usd.keys())}"
    en_usd = monto * tasas_a_usd[de_upper]
    resultado = en_usd / tasas_a_usd[a_upper]
    return f"{monto} {de_upper} = {resultado:.2f} {a_upper}"


@tool
def listar_archivos(directorio: str = ".") -> str:
    """Lista los archivos Python en el directorio actual del proyecto."""
    import os
    try:
        archivos = [f for f in os.listdir(directorio) if f.endswith(".py")]
        archivos.sort()
        return f"Archivos .py en '{directorio}': {archivos[:10]}"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [obtener_clima, convertir_moneda, listar_archivos]


# ── Tool calling a bajo nivel ─────────────────────────────────────────────────

def ejecutar_tool_call(tool_call: dict) -> str:
    """Ejecuta un tool_call generado por el LLM y retorna el resultado."""
    nombre = tool_call["name"]
    args = tool_call["args"]

    tool_map = {t.name: t for t in TOOLS}
    if nombre not in tool_map:
        return f"Herramienta '{nombre}' no encontrada"

    return str(tool_map[nombre].invoke(args))


def ciclo_tool_calling(pregunta: str) -> list[BaseMessage]:
    """
    Implementación manual del ciclo de tool calling:
      1. LLM con herramientas enlazadas
      2. Si pide tool → ejecutar → devolver resultado → repetir
      3. Si responde directamente → fin
    """
    llm = get_llm()
    llm_con_tools = llm.bind_tools(TOOLS)

    mensajes: list[BaseMessage] = [HumanMessage(content=pregunta)]
    max_pasos = 5  # evitar loops infinitos

    for paso in range(max_pasos):
        respuesta: AIMessage = llm_con_tools.invoke(mensajes)
        mensajes.append(respuesta)

        console.print(f"  [dim]Paso {paso + 1}:[/]", end=" ")

        if not respuesta.tool_calls:
            console.print("[green]Respuesta final (sin más herramientas)[/]")
            break

        # El LLM pidió herramientas
        for tc in respuesta.tool_calls:
            console.print(f"[yellow]tool_call → {tc['name']}({tc['args']})[/]")
            resultado = ejecutar_tool_call(tc)
            console.print(f"  [cyan]resultado:[/] {resultado}")

            # Agregar el resultado como ToolMessage
            mensajes.append(ToolMessage(
                content=resultado,
                tool_call_id=tc["id"],
            ))

    return mensajes


# ── Inspeccionar el schema de una herramienta ─────────────────────────────────

def mostrar_schemas():
    console.print(Panel(
        "[bold]Lo que el LLM ve de cada herramienta:[/]\n\n"
        "[dim]El LLM recibe el nombre, descripción y schema JSON de cada tool.\n"
        "Con eso decide cuándo llamarla y con qué argumentos.[/]",
        border_style="blue",
    ))

    for t in TOOLS:
        schema = t.args_schema.schema() if t.args_schema else {}
        console.print(Panel(
            Syntax(json.dumps(schema, indent=2, ensure_ascii=False), "json", theme="monokai"),
            title=f"[bold]{t.name}[/] — {t.description[:60]}",
            border_style="cyan",
        ))


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 9.2: Tool Calling")

    console.print(Panel(
        "[bold]Flujo de tool calling:[/]\n\n"
        "  1. [cyan]bind_tools()[/]      enlaza herramientas al LLM\n"
        "  2. LLM invocado → puede generar [cyan]AIMessage.tool_calls[/]\n"
        "  3. Tu código ejecuta la herramienta\n"
        "  4. Resultado como [cyan]ToolMessage[/] → de vuelta al LLM\n"
        "  5. LLM genera respuesta final\n\n"
        "[dim]El LLM no ejecuta código — solo genera JSON con nombre y args.\n"
        "TÚ ejecutas la herramienta y le devuelves el resultado.[/]",
        border_style="blue",
        title="Tool Calling",
    ))

    mostrar_schemas()

    preguntas = [
        "¿Qué clima hace en Madrid?",
        "¿Cuánto son 100 EUR en pesos mexicanos?",
        "¿Qué clima hace en Barcelona y cuánto son 50 USD en ARS?",
        "¿Cuáles son los archivos Python del proyecto?",
    ]

    for pregunta in preguntas:
        console.rule(f"[bold]Pregunta: {pregunta}")
        mensajes = ciclo_tool_calling(pregunta)
        respuesta_final = mensajes[-1].content
        console.print(Panel(
            respuesta_final,
            title="Respuesta final",
            border_style="green",
        ))

    # Mostrar el AIMessage raw con tool_calls
    console.rule("[yellow]Estructura interna: AIMessage con tool_calls")
    llm = get_llm().bind_tools(TOOLS)
    msg = llm.invoke([HumanMessage(content="¿Qué clima hace en Buenos Aires?")])
    if msg.tool_calls:
        console.print(Panel(
            Syntax(json.dumps(msg.tool_calls, indent=2, ensure_ascii=False), "json", theme="monokai"),
            title="AIMessage.tool_calls (lo que genera el LLM)",
            border_style="magenta",
        ))


if __name__ == "__main__":
    main()
