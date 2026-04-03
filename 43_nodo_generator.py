"""
43_nodo_generator.py — Módulo 8.5: Nodo Generator — genera respuesta con contexto

El generator es el nodo que finalmente responde al usuario.
Recibe del estado:
  - pregunta: la pregunta original (o reformulada)
  - documentos_relevantes: chunks filtrados por el Grader

Variaciones cubiertas:
  - Generator básico (prompt + LLM)
  - Generator con citación de fuentes
  - Generator con control de confianza (sabe cuándo no sabe)
"""
import logging
from typing import TypedDict, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm, format_docs
from rag.retriever import get_retriever, get_vectorstore
from rag.config import RETRIEVAL_K

console = Console()


# ── Estado ────────────────────────────────────────────────────────────────────

class EstadoGenerator(TypedDict):
    pregunta: str
    documentos: list[Document]
    contexto: str
    respuesta: str
    fuentes_citadas: list[str]
    confianza: str          # "alta", "media", "baja"
    logs: Annotated[list[str], operator.add]


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_BASICO = ChatPromptTemplate.from_template(
    """Eres un asistente útil. Responde SOLO basándote en el contexto proporcionado.
Si la respuesta no está en el contexto, di "No tengo suficiente información."

Contexto:
{context}

Pregunta: {question}

Respuesta (en español, clara y concisa):"""
)

PROMPT_CON_FUENTES = ChatPromptTemplate.from_template(
    """Eres un asistente útil. Responde basándote en el contexto.
Al final de tu respuesta, lista las fuentes usadas.

Contexto:
{context}

Fuentes disponibles: {fuentes}

Pregunta: {question}

Respuesta (en español):
[Respuesta aquí]

Fuentes consultadas:
[Lista aquí solo las fuentes que usaste]"""
)

PROMPT_CON_CONFIANZA = ChatPromptTemplate.from_template(
    """Eres un asistente preciso. Responde la pregunta y evalúa tu confianza.

Contexto:
{context}

Pregunta: {question}

Responde en este formato exacto:
RESPUESTA: [tu respuesta]
CONFIANZA: [alta|media|baja] — [razón en una frase]"""
)


# ── Nodo Retriever auxiliar ───────────────────────────────────────────────────

def nodo_retriever(estado: EstadoGenerator) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])
    return {"documentos": docs, "logs": [f"retriever: {len(docs)} docs"]}


# ── Nodo Generator — básico ───────────────────────────────────────────────────

def nodo_generator_basico(estado: EstadoGenerator) -> dict:
    contexto = format_docs(estado["documentos"])
    llm = get_llm()
    respuesta = (PROMPT_BASICO | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content

    return {
        "contexto": contexto,
        "respuesta": respuesta,
        "fuentes_citadas": [],
        "confianza": "media",
        "logs": [f"generator_basico: {len(respuesta)} chars"],
    }


# ── Nodo Generator — con fuentes ─────────────────────────────────────────────

def nodo_generator_con_fuentes(estado: EstadoGenerator) -> dict:
    contexto = format_docs(estado["documentos"])
    fuentes = list({
        doc.metadata.get("source", "desconocido")
        for doc in estado["documentos"]
    })

    llm = get_llm()
    respuesta_completa = (PROMPT_CON_FUENTES | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
        "fuentes": ", ".join(fuentes),
    }).content

    return {
        "contexto": contexto,
        "respuesta": respuesta_completa,
        "fuentes_citadas": fuentes,
        "confianza": "media",
        "logs": [f"generator_fuentes: {len(fuentes)} fuentes citadas"],
    }


# ── Nodo Generator — con confianza ───────────────────────────────────────────

def nodo_generator_con_confianza(estado: EstadoGenerator) -> dict:
    contexto = format_docs(estado["documentos"])
    llm = get_llm()
    respuesta_raw = (PROMPT_CON_CONFIANZA | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content

    # Parsear respuesta y confianza
    respuesta = respuesta_raw
    confianza = "media"

    for linea in respuesta_raw.split("\n"):
        if linea.startswith("RESPUESTA:"):
            respuesta = linea.replace("RESPUESTA:", "").strip()
        elif linea.startswith("CONFIANZA:"):
            texto_confianza = linea.replace("CONFIANZA:", "").strip().lower()
            if "alta" in texto_confianza:
                confianza = "alta"
            elif "baja" in texto_confianza:
                confianza = "baja"

    return {
        "contexto": contexto,
        "respuesta": respuesta,
        "fuentes_citadas": [],
        "confianza": confianza,
        "logs": [f"generator_confianza: nivel={confianza}"],
    }


# ── Grafo ─────────────────────────────────────────────────────────────────────

def construir_grafo(modo: str = "basico"):
    builder = StateGraph(EstadoGenerator)

    modos = {
        "basico": nodo_generator_basico,
        "fuentes": nodo_generator_con_fuentes,
        "confianza": nodo_generator_con_confianza,
    }
    generator_fn = modos.get(modo, nodo_generator_basico)

    builder.add_node("retriever", nodo_retriever)
    builder.add_node("generator", generator_fn)

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 8.5: Nodo Generator")

    console.print(Panel(
        "[bold]Variaciones del Generator:[/]\n\n"
        "  [cyan]básico[/]:      responde con el contexto\n"
        "  [cyan]con_fuentes[/]: incluye las fuentes consultadas\n"
        "  [cyan]confianza[/]:   evalúa su propia confianza en la respuesta\n\n"
        "[dim]El módulo 8.6 (Hallucination Check) agrega una capa de\n"
        "verificación DESPUÉS del generator.[/]",
        border_style="blue",
        title="Nodo Generator",
    ))

    pregunta = "¿De qué trata el documento?"

    estado_base: EstadoGenerator = {
        "pregunta": pregunta,
        "documentos": [],
        "contexto": "",
        "respuesta": "",
        "fuentes_citadas": [],
        "confianza": "",
        "logs": [],
    }

    tabla = Table(title="Comparación de modos del Generator", show_header=True, header_style="bold magenta")
    tabla.add_column("Modo")
    tabla.add_column("Fuentes citadas")
    tabla.add_column("Confianza")
    tabla.add_column("Respuesta (80 chars)")

    for modo in ["basico", "fuentes", "confianza"]:
        grafo = construir_grafo(modo)
        resultado = grafo.invoke(estado_base.copy())
        tabla.add_row(
            modo,
            str(resultado["fuentes_citadas"])[:30] or "—",
            resultado["confianza"],
            resultado["respuesta"][:80].replace("\n", " ") + "...",
        )

    console.print(tabla)


if __name__ == "__main__":
    main()
