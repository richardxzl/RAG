"""
50_human_in_the_loop.py — Módulo 9.5: Human-in-the-loop

Human-in-the-loop (HITL) permite pausar la ejecución del grafo,
mostrar el estado actual al usuario, y continuar según su respuesta.

Casos de uso:
  - Confirmar antes de ejecutar una acción irreversible
  - Pedir clarificación cuando la intención es ambigua
  - Revisar una respuesta antes de enviarla
  - Aprobar el uso de una herramienta costosa o sensible

Componentes necesarios:
  1. interrupt():    pausa el grafo y devuelve control al caller
  2. MemorySaver:    checkpointer en memoria para persistir el estado entre pausas
  3. thread_config:  identificador de la conversación pausada
  4. Command(resume): continúa la ejecución desde donde se pausó
"""
import logging
from typing import TypedDict, Annotated, Literal

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, create_react_agent

from rag.chain import get_llm

console = Console()


# ── Herramientas que requieren confirmación ───────────────────────────────────

@tool
def enviar_email(destinatario: str, asunto: str, cuerpo: str) -> str:
    """Envía un email. REQUIERE CONFIRMACIÓN DEL USUARIO antes de ejecutar."""
    # Simulado — en producción llamaría a la API de email real
    console.print(f"  [green]✉ Email enviado a {destinatario}[/]")
    return f"Email enviado a {destinatario} con asunto '{asunto}'"


@tool
def eliminar_archivo(ruta: str) -> str:
    """Elimina un archivo del sistema. REQUIERE CONFIRMACIÓN DEL USUARIO."""
    # Simulado
    console.print(f"  [red]🗑 Archivo eliminado: {ruta}[/]")
    return f"Archivo '{ruta}' eliminado"


@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática. No requiere confirmación."""
    try:
        resultado = eval(expresion, {"__builtins__": {}}, {})
        return f"{expresion} = {resultado}"
    except Exception as e:
        return f"Error: {e}"


# ── Patrón 1: interrupt() antes de ejecutar una tool ─────────────────────────
# Este es el patrón más común: el agente propone una acción, nosotros la aprobamos.

TOOLS_PELIGROSAS = {"enviar_email", "eliminar_archivo"}


class EstadoHITL(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]


def nodo_llm(estado: EstadoHITL) -> dict:
    llm = get_llm()
    tools = [enviar_email, eliminar_archivo, calcular]
    llm_con_tools = llm.bind_tools(tools)
    respuesta = llm_con_tools.invoke(estado["messages"])
    return {"messages": [respuesta]}


def nodo_tools_con_confirmacion(estado: EstadoHITL) -> dict:
    """
    Ejecuta las tools, pero pausa antes de las peligrosas para pedir confirmación.
    interrupt() congela el grafo aquí y devuelve control al caller.
    """
    ultimo_msg = estado["messages"][-1]
    if not isinstance(ultimo_msg, AIMessage) or not ultimo_msg.tool_calls:
        return {"messages": []}

    resultados = []
    for tc in ultimo_msg.tool_calls:
        if tc["name"] in TOOLS_PELIGROSAS:
            # ── PAUSA: pedir confirmación ─────────────────────────────────────
            # interrupt() devuelve el valor que se pase a Command(resume=valor)
            aprobacion = interrupt({
                "tipo": "confirmacion",
                "accion": tc["name"],
                "args": tc["args"],
                "mensaje": f"El agente quiere ejecutar '{tc['name']}' con args {tc['args']}. ¿Aprobar?",
            })

            if aprobacion.lower() not in ("s", "si", "sí", "yes", "y"):
                resultados.append(ToolMessage(
                    content=f"Acción '{tc['name']}' cancelada por el usuario.",
                    tool_call_id=tc["id"],
                ))
                continue

        # Ejecutar la tool (peligrosa aprobada, o no peligrosa)
        tool_map = {t.name: t for t in [enviar_email, eliminar_archivo, calcular]}
        if tc["name"] in tool_map:
            resultado = str(tool_map[tc["name"]].invoke(tc["args"]))
        else:
            resultado = f"Tool '{tc['name']}' no encontrada"

        resultados.append(ToolMessage(content=resultado, tool_call_id=tc["id"]))

    return {"messages": resultados}


def necesita_tools(estado: EstadoHITL) -> str:
    ultimo = estado["messages"][-1]
    if isinstance(ultimo, AIMessage) and ultimo.tool_calls:
        return "tools"
    return END


def construir_grafo_hitl():
    """Grafo ReAct con human-in-the-loop en el nodo de tools."""
    checkpointer = MemorySaver()

    builder = StateGraph(EstadoHITL)
    builder.add_node("llm", nodo_llm)
    builder.add_node("tools", nodo_tools_con_confirmacion)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", necesita_tools, {
        "tools": "tools",
        END: END,
    })
    builder.add_edge("tools", "llm")

    # El checkpointer es OBLIGATORIO para HITL — persiste el estado pausado
    return builder.compile(checkpointer=checkpointer)


# ── Patrón 2: HITL con create_react_agent (más simple) ───────────────────────

def construir_agente_con_hitl():
    """
    create_react_agent también soporta HITL via interrupt_before.
    Pausa ANTES de ejecutar cualquier tool.
    """
    llm = get_llm()
    tools = [enviar_email, calcular]
    checkpointer = MemorySaver()

    return create_react_agent(
        llm,
        tools,
        checkpointer=checkpointer,
        interrupt_before=["tools"],  # pausa antes de ejecutar cualquier tool
    )


# ── Demo interactivo ──────────────────────────────────────────────────────────

def demo_hitl_interactivo():
    """
    Demo completo con interacción real del usuario.
    El grafo se pausa, el usuario ve la acción propuesta y decide.
    """
    console.print(Panel(
        "[bold]Demo interactivo:[/]\n"
        "El agente propondrá acciones. Puedes aprobarlas (s) o rechazarlas (n).",
        border_style="blue",
    ))

    grafo = construir_grafo_hitl()
    thread_config = {"configurable": {"thread_id": "demo-hitl-1"}}

    pregunta = "Calcula 15 * 8 y luego envía un email a test@ejemplo.com con el resultado"
    console.print(f"\n[bold]Pregunta:[/] {pregunta}\n")

    # Primera ejecución — se pausará en interrupt()
    resultado = grafo.invoke(
        {"messages": [HumanMessage(content=pregunta)]},
        config=thread_config,
    )

    # Procesar interrupciones hasta que el grafo termine
    while True:
        estado_actual = grafo.get_state(thread_config)

        # Buscar si hay una interrupción pendiente
        if not estado_actual.tasks:
            break

        interrupcion = None
        for task in estado_actual.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupcion = task.interrupts[0].value
                break

        if not interrupcion:
            break

        # Mostrar la acción propuesta al usuario
        console.print(Panel(
            f"[bold yellow]Acción propuesta:[/]\n"
            f"  Tool: [cyan]{interrupcion['accion']}[/]\n"
            f"  Args: {interrupcion['args']}\n\n"
            f"{interrupcion['mensaje']}",
            border_style="yellow",
            title="⚠ Confirmación requerida",
        ))

        respuesta = Prompt.ask("[bold]¿Aprobar?[/] (s/n)")

        # Reanudar el grafo con la respuesta del usuario
        resultado = grafo.invoke(
            Command(resume=respuesta),
            config=thread_config,
        )

    # Respuesta final
    if resultado and "messages" in resultado:
        console.print(Panel(
            resultado["messages"][-1].content,
            title="Respuesta final del agente",
            border_style="green",
        ))


# ── Demo sin interacción (auto-aprobación para CI) ────────────────────────────

def demo_hitl_auto():
    """
    Demo sin interacción: aprueba todo automáticamente.
    Útil para tests y para entender el flujo sin input manual.
    """
    console.print(Panel(
        "[bold]Demo automático (aprueba todo):[/]\n"
        "Simula el flujo HITL sin requerir input del usuario.",
        border_style="cyan",
    ))

    grafo = construir_grafo_hitl()
    thread_config = {"configurable": {"thread_id": "demo-auto-1"}}

    pregunta = "Calcula 7 * 6 y envía los resultados por email a resumen@empresa.com"
    console.print(f"\n[bold]Pregunta:[/] {pregunta}\n")

    grafo.invoke(
        {"messages": [HumanMessage(content=pregunta)]},
        config=thread_config,
    )

    iteraciones = 0
    max_iter = 10

    while iteraciones < max_iter:
        iteraciones += 1
        estado_actual = grafo.get_state(thread_config)

        if not estado_actual.tasks:
            break

        interrupcion = None
        for task in estado_actual.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupcion = task.interrupts[0].value
                break

        if not interrupcion:
            break

        console.print(Panel(
            f"[yellow]Confirmación automática:[/]\n"
            f"  Tool: [cyan]{interrupcion.get('accion', '?')}[/]\n"
            f"  Args: {interrupcion.get('args', {})}",
            border_style="yellow",
        ))

        # Auto-aprobar
        resultado = grafo.invoke(
            Command(resume="s"),
            config=thread_config,
        )

    estado_final = grafo.get_state(thread_config)
    if estado_final.values.get("messages"):
        ultimo = estado_final.values["messages"][-1]
        console.print(Panel(
            ultimo.content,
            title="Respuesta final",
            border_style="green",
        ))


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 9.5: Human-in-the-Loop")

    console.print(Panel(
        "[bold]¿Por qué HITL?[/]\n\n"
        "Algunos agentes tienen herramientas que pueden:\n"
        "  • Enviar emails o mensajes\n"
        "  • Borrar datos\n"
        "  • Hacer compras o transacciones\n"
        "  • Ejecutar código en producción\n\n"
        "HITL permite pausar antes de esas acciones y pedir aprobación.\n\n"
        "[bold]Componentes:[/]\n"
        "  [cyan]interrupt(value)[/]   → pausa el grafo, retorna 'value' al caller\n"
        "  [cyan]MemorySaver()[/]       → persiste el estado pausado\n"
        "  [cyan]Command(resume=v)[/]   → reanuda el grafo con valor 'v'\n"
        "  [cyan]thread_id[/]          → identifica la conversación pausada",
        border_style="blue",
        title="Human-in-the-Loop",
    ))

    # Mostrar el grafo
    grafo = construir_grafo_hitl()
    console.print(Panel(
        grafo.get_graph().draw_ascii(),
        title="Grafo con HITL",
        border_style="yellow",
    ))

    # Comparar interrupt_before vs interrupt() manual
    tabla = Table(
        title="Estrategias de HITL",
        show_header=True,
        header_style="bold magenta",
    )
    tabla.add_column("Estrategia", style="bold")
    tabla.add_column("Cómo")
    tabla.add_column("Cuándo usar")

    tabla.add_row(
        "interrupt_before=['tools']",
        "create_react_agent(..., interrupt_before=['tools'])",
        "Confirmar CUALQUIER herramienta",
    )
    tabla.add_row(
        "interrupt() manual",
        "interrupt(datos) dentro del nodo",
        "Lógica selectiva (solo tools peligrosas)",
    )
    tabla.add_row(
        "Nodo de revisión",
        "Nodo intermedio que pausa siempre",
        "Aprobar respuestas antes de enviarlas",
    )
    console.print(tabla)

    # Ejecutar demo
    console.rule("[bold]Ejecutando demo automático")
    demo_hitl_auto()

    # Preguntar si se quiere demo interactivo
    console.print(Panel(
        "Para el demo interactivo (requiere input manual), ejecuta:\n\n"
        "[cyan]  demo_hitl_interactivo()[/]\n\n"
        "El script pausará y te pedirá aprobar/rechazar cada acción.",
        border_style="dim",
    ))


if __name__ == "__main__":
    main()
