"""
36_edges.py — Módulo 7.3: Edges — condicionales y rutas entre nodos

Hay tres tipos de edges en LangGraph:

  1. Edge directa:      add_edge("nodo_a", "nodo_b")
                        → siempre va de A a B

  2. Edge condicional:  add_conditional_edges("nodo_a", funcion_router, mapa)
                        → la función decide a qué nodo ir según el estado

  3. Edge especial:     START y END como origen/destino

La función de routing:
  - Recibe el estado
  - Retorna un STRING que es una key del mapa de destinos
  - NO modifica el estado (solo lee)
"""
import logging
from typing import TypedDict, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langgraph.graph import StateGraph, START, END

console = Console()


# ── Estado ────────────────────────────────────────────────────────────────────

class EstadoRouter(TypedDict):
    pregunta: str
    tipo: str        # "matematica", "historia", "general"
    respuesta: str
    ruta_tomada: str


# ── Nodos ─────────────────────────────────────────────────────────────────────

def clasificar(estado: EstadoRouter) -> dict:
    """Clasifica la pregunta para que el router sepa qué rama tomar."""
    pregunta = estado["pregunta"].lower()
    if any(w in pregunta for w in ["suma", "resta", "cuanto es", "calcul", "número"]):
        tipo = "matematica"
    elif any(w in pregunta for w in ["historia", "año", "guerra", "siglo", "cuando"]):
        tipo = "historia"
    else:
        tipo = "general"

    return {"tipo": tipo}


def responder_matematica(estado: EstadoRouter) -> dict:
    return {
        "respuesta": f"[Matemática] Analizando: {estado['pregunta']}",
        "ruta_tomada": "matematica",
    }


def responder_historia(estado: EstadoRouter) -> dict:
    return {
        "respuesta": f"[Historia] Contextualizando: {estado['pregunta']}",
        "ruta_tomada": "historia",
    }


def responder_general(estado: EstadoRouter) -> dict:
    return {
        "respuesta": f"[General] Respondiendo: {estado['pregunta']}",
        "ruta_tomada": "general",
    }


# ── Router (función de routing) ───────────────────────────────────────────────
# Esta función NO es un nodo — es la función que decide el edge condicional.
# Retorna un string que mapea al siguiente nodo.

def router(estado: EstadoRouter) -> Literal["matematica", "historia", "general"]:
    """
    Lee 'tipo' del estado y retorna el nombre del nodo destino.
    El tipo ya fue determinado por el nodo 'clasificar'.
    """
    return estado["tipo"]


# ── Grafo con edges condicionales ─────────────────────────────────────────────

def construir_grafo():
    builder = StateGraph(EstadoRouter)

    # Nodos
    builder.add_node("clasificar", clasificar)
    builder.add_node("matematica", responder_matematica)
    builder.add_node("historia", responder_historia)
    builder.add_node("general", responder_general)

    # Edges directas
    builder.add_edge(START, "clasificar")

    # Edge condicional: después de clasificar, el router decide el camino
    builder.add_conditional_edges(
        "clasificar",   # nodo de origen
        router,         # función que retorna un string
        {               # mapa: string → nodo destino
            "matematica": "matematica",
            "historia": "historia",
            "general": "general",
        },
    )

    # Todos los nodos terminales van a END
    builder.add_edge("matematica", END)
    builder.add_edge("historia", END)
    builder.add_edge("general", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 7.3: Edges condicionales")

    console.print(Panel(
        "[bold]Tipos de edges:[/]\n\n"
        "  [cyan]add_edge(A, B)[/]               → siempre va de A a B\n"
        "  [cyan]add_conditional_edges(A, fn, mapa)[/] → fn decide el destino\n\n"
        "[bold]La función de routing:[/]\n"
        "  - Recibe el estado completo\n"
        "  - Retorna un string (key del mapa)\n"
        "  - NO modifica el estado\n\n"
        "[dim]Patrón típico: un nodo clasifica → otro nodo routea → n ramas[/]",
        border_style="blue",
        title="Edges en LangGraph",
    ))

    grafo = construir_grafo()

    preguntas = [
        "¿Cuánto es 5 más 3?",
        "¿En qué año terminó la Segunda Guerra Mundial?",
        "¿Qué es LangChain?",
        "Calcula el área de un triángulo",
        "¿Cuándo fue la Revolución Francesa?",
    ]

    tabla = Table(title="Routing de preguntas", show_header=True, header_style="bold magenta")
    tabla.add_column("Pregunta", max_width=45)
    tabla.add_column("Tipo detectado")
    tabla.add_column("Ruta tomada")

    for pregunta in preguntas:
        estado_inicial: EstadoRouter = {
            "pregunta": pregunta,
            "tipo": "",
            "respuesta": "",
            "ruta_tomada": "",
        }
        resultado = grafo.invoke(estado_inicial)
        tabla.add_row(pregunta[:44], resultado["tipo"], resultado["ruta_tomada"])

    console.print(tabla)

    # Explicar la diferencia entre nodo router y función router
    console.print(Panel(
        "[bold]¿Nodo clasificador vs función router?[/]\n\n"
        "  [cyan]Nodo clasificar[/]:  MODIFICA el estado (escribe 'tipo')\n"
        "  [cyan]Función router[/]:   SOLO LEE el estado, retorna string\n\n"
        "  El patrón clásico:\n"
        "    1. Nodo A hace el trabajo (llama LLM, etc.)\n"
        "    2. Función router inspecciona el resultado\n"
        "    3. Edge condicional elige el siguiente nodo\n\n"
        "[dim]La función router puede también llamar al LLM directamente\n"
        "si necesita razonamiento para decidir la ruta.[/]",
        border_style="yellow",
    ))


if __name__ == "__main__":
    main()
