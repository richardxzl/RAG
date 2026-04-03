"""
40_nodo_router.py — Módulo 8.2: Nodo Router — ¿RAG o conversacional?

No toda pregunta necesita buscar en documentos. Un router inteligente
decide el camino antes de desperdiciar recursos en el retriever.

Dos estrategias:
  1. Router basado en keywords (rápido, sin LLM)
  2. Router con LLM (preciso, más lento)

Grafo:
  START → router → [retriever → generator] o [directo] → END
"""
import logging
from typing import TypedDict, Literal, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Estado ────────────────────────────────────────────────────────────────────

class EstadoConRouter(TypedDict):
    pregunta: str
    ruta: str                       # "rag" o "conversacional"
    documentos: list[Document]
    contexto: str
    respuesta: str
    logs: Annotated[list[str], operator.add]


# ── Estrategia 1: Router por keywords (sin LLM) ───────────────────────────────

KEYWORDS_RAG = [
    "documento", "archivo", "texto", "según", "menciona", "dice",
    "trata", "habla", "contenido", "información del", "qué dice",
]

KEYWORDS_CONVERSACIONAL = [
    "hola", "gracias", "cómo estás", "bye", "adiós", "quién eres",
    "qué eres", "puedes ayudarme", "qué puedes hacer",
]


def nodo_router_keywords(estado: EstadoConRouter) -> dict:
    """
    Router rápido basado en keywords.
    No llama al LLM → latencia cero, pero menos preciso.
    """
    pregunta = estado["pregunta"].lower()

    if any(k in pregunta for k in KEYWORDS_CONVERSACIONAL):
        ruta = "conversacional"
    elif any(k in pregunta for k in KEYWORDS_RAG):
        ruta = "rag"
    else:
        # Default: intentar RAG si no está claro
        ruta = "rag"

    console.print(f"  [yellow]router (keywords):[/] '{pregunta[:40]}' → {ruta}")
    return {
        "ruta": ruta,
        "logs": [f"router_keywords: '{pregunta[:30]}' → {ruta}"],
    }


# ── Estrategia 2: Router con LLM ──────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """Clasifica la siguiente pregunta. Responde SOLO con una palabra.

Pregunta: {pregunta}

Si la pregunta es sobre el contenido de un documento específico, archivo o texto → responde: rag
Si la pregunta es una conversación general (saludo, agradecimiento, pregunta genérica) → responde: conversacional

Respuesta (solo una palabra):"""
)


def nodo_router_llm(estado: EstadoConRouter) -> dict:
    """
    Router preciso usando el LLM.
    Más costoso pero entiende contexto y variaciones del lenguaje.
    """
    llm = get_llm()
    cadena = ROUTER_PROMPT | llm
    resultado = cadena.invoke({"pregunta": estado["pregunta"]}).content.strip().lower()

    # Normalizar por si el LLM agrega texto extra
    ruta = "rag" if "rag" in resultado else "conversacional"

    console.print(f"  [yellow]router (LLM):[/] respuesta raw='{resultado}' → {ruta}")
    return {
        "ruta": ruta,
        "logs": [f"router_llm: '{estado['pregunta'][:30]}' → {ruta}"],
    }


# ── Nodos de respuesta ────────────────────────────────────────────────────────

def nodo_retriever(estado: EstadoConRouter) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])
    return {
        "documentos": docs,
        "logs": [f"retriever: {len(docs)} docs"],
    }


def nodo_generator(estado: EstadoConRouter) -> dict:
    contexto = format_docs(estado["documentos"])
    llm = get_llm()
    respuesta = (QUERY_PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content
    return {
        "contexto": contexto,
        "respuesta": respuesta,
        "logs": ["generator: respuesta con RAG"],
    }


def nodo_directo(estado: EstadoConRouter) -> dict:
    """Responde sin buscar en documentos."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente conversacional amigable. Responde en español."),
        ("human", "{pregunta}"),
    ])
    respuesta = (prompt | llm).invoke({"pregunta": estado["pregunta"]}).content
    return {
        "respuesta": respuesta,
        "logs": ["directo: respuesta conversacional"],
    }


# ── Función de routing (edge condicional) ─────────────────────────────────────

def decidir_ruta(estado: EstadoConRouter) -> Literal["retriever", "directo"]:
    return "retriever" if estado["ruta"] == "rag" else "directo"


# ── Grafo ─────────────────────────────────────────────────────────────────────

def construir_grafo(usar_llm: bool = False):
    builder = StateGraph(EstadoConRouter)

    router_fn = nodo_router_llm if usar_llm else nodo_router_keywords

    builder.add_node("router", router_fn)
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
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 8.2: Nodo Router")

    console.print(Panel(
        "[bold]¿Por qué un router?[/]\n\n"
        "Sin router: cada pregunta va al retriever, aunque no necesite documentos.\n"
        "Con router:  saludos y preguntas generales van directamente al LLM.\n\n"
        "  [yellow]keywords[/]: rápido, sin costo extra de LLM, menos preciso\n"
        "  [yellow]LLM[/]:      preciso, entiende variaciones, +latencia y costo",
        border_style="blue",
        title="Nodo Router",
    ))

    preguntas = [
        ("¿Qué dice el documento sobre Python?", "esperado: rag"),
        ("Hola, ¿cómo estás?", "esperado: conversacional"),
        ("¿De qué trata el archivo?", "esperado: rag"),
        ("Gracias por tu ayuda", "esperado: conversacional"),
        ("¿Cuál es el tema principal del texto?", "esperado: rag"),
    ]

    # Demo con router de keywords
    console.rule("[yellow]Router por keywords")
    grafo_kw = construir_grafo(usar_llm=False)

    tabla = Table(show_header=True, header_style="bold magenta")
    tabla.add_column("Pregunta", max_width=45)
    tabla.add_column("Nota")
    tabla.add_column("Ruta asignada")
    tabla.add_column("Respuesta (40 chars)")

    for pregunta, nota in preguntas:
        estado: EstadoConRouter = {
            "pregunta": pregunta,
            "ruta": "",
            "documentos": [],
            "contexto": "",
            "respuesta": "",
            "logs": [],
        }
        resultado = grafo_kw.invoke(estado)
        tabla.add_row(
            pregunta[:44],
            nota,
            resultado["ruta"],
            resultado["respuesta"][:40] + "...",
        )

    console.print(tabla)


if __name__ == "__main__":
    main()
