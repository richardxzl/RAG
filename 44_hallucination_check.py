"""
44_hallucination_check.py — Módulo 8.6: Nodo Hallucination Check

Una alucinación ocurre cuando el LLM genera información que NO está
en los documentos recuperados. El Hallucination Check verifica esto.

Patrón: generator → hallucination_check → [si OK: END] [si alucina: regenerar]

¿Cómo detectar alucinaciones?
  1. LLM judge: le preguntamos al mismo LLM si la respuesta es fiel al contexto
  2. Keyword overlap: verificación rápida sin LLM (menos precisa)

En este módulo implementamos el LLM judge, que es el estándar en producción.
"""
from typing import TypedDict, Literal, Annotated
import operator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()

MAX_REGENERACIONES = 2


# ── Estado ────────────────────────────────────────────────────────────────────

class EstadoHallucination(TypedDict):
    pregunta: str
    documentos: list[Document]
    contexto: str
    respuesta: str
    es_fiel: bool           # ¿la respuesta está respaldada por los docs?
    razon_fidelidad: str    # explicación del juicio
    regeneraciones: int
    logs: Annotated[list[str], operator.add]


# ── Nodo Retriever ────────────────────────────────────────────────────────────

def nodo_retriever(estado: EstadoHallucination) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])
    return {"documentos": docs, "logs": [f"retriever: {len(docs)} docs"]}


# ── Nodo Generator ────────────────────────────────────────────────────────────

def nodo_generator(estado: EstadoHallucination) -> dict:
    contexto = format_docs(estado["documentos"])
    llm = get_llm()
    respuesta = (QUERY_PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content
    return {
        "contexto": contexto,
        "respuesta": respuesta,
        "regeneraciones": estado["regeneraciones"] + 1,
        "logs": [f"generator: intento #{estado['regeneraciones'] + 1}"],
    }


# ── Nodo Hallucination Check ──────────────────────────────────────────────────

HALLUCINATION_PROMPT = ChatPromptTemplate.from_template(
    """Eres un evaluador de calidad para sistemas RAG.

Tu tarea: determinar si la RESPUESTA está completamente respaldada por el CONTEXTO.

CONTEXTO (fuente de verdad):
{contexto}

RESPUESTA GENERADA:
{respuesta}

Evalúa:
- ¿Cada afirmación de la respuesta se puede verificar en el contexto?
- ¿La respuesta inventa datos, fechas, nombres o hechos que no están en el contexto?

Responde en este formato exacto:
VEREDICTO: fiel  (si TODO está en el contexto)
VEREDICTO: alucinacion  (si hay algo inventado)
RAZON: [explica en una oración qué falta o qué está bien]"""
)


def nodo_hallucination_check(estado: EstadoHallucination) -> dict:
    """
    Evalúa si la respuesta generada está completamente respaldada
    por los documentos recuperados.
    """
    llm = get_llm()
    evaluacion = (HALLUCINATION_PROMPT | llm).invoke({
        "contexto": estado["contexto"][:2000],  # limitar tokens
        "respuesta": estado["respuesta"],
    }).content

    # Parsear veredicto
    es_fiel = True
    razon = "no evaluado"

    for linea in evaluacion.split("\n"):
        linea = linea.strip()
        if linea.startswith("VEREDICTO:"):
            veredicto = linea.replace("VEREDICTO:", "").strip().lower()
            es_fiel = "alucinacion" not in veredicto
        elif linea.startswith("RAZON:"):
            razon = linea.replace("RAZON:", "").strip()

    estado_visual = "[green]FIEL[/]" if es_fiel else "[red]ALUCINACIÓN DETECTADA[/]"
    console.print(f"  [cyan]hallucination_check:[/] {estado_visual}")
    console.print(f"  Razón: {razon}")

    return {
        "es_fiel": es_fiel,
        "razon_fidelidad": razon,
        "logs": [f"hallucination_check: fiel={es_fiel}, razón='{razon[:60]}'"],
    }


# ── Routing post-hallucination check ─────────────────────────────────────────

def decidir_post_check(
    estado: EstadoHallucination,
) -> Literal["generator", "__end__"]:
    if estado["es_fiel"]:
        console.print("  [green]✓ Respuesta validada[/]")
        return END
    if estado["regeneraciones"] >= MAX_REGENERACIONES:
        console.print(
            f"  [yellow]Máximo de regeneraciones ({MAX_REGENERACIONES}) alcanzado[/]"
        )
        return END
    console.print(
        f"  [yellow]Regenerando... (intento {estado['regeneraciones'] + 1})[/]"
    )
    return "generator"


# ── Grafo ─────────────────────────────────────────────────────────────────────

def construir_grafo():
    builder = StateGraph(EstadoHallucination)

    builder.add_node("retriever", nodo_retriever)
    builder.add_node("generator", nodo_generator)
    builder.add_node("hallucination_check", nodo_hallucination_check)

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", "hallucination_check")
    builder.add_conditional_edges(
        "hallucination_check",
        decidir_post_check,
        {
            "generator": "generator",
            END: END,
        },
    )

    return builder.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]RAG Lab — Módulo 8.6: Hallucination Check")

    console.print(Panel(
        "[bold]¿Qué es una alucinación en RAG?[/]\n\n"
        "El LLM genera algo que NO está en los documentos recuperados.\n"
        "Puede ser:\n"
        "  • Un dato inventado (fecha, número, nombre)\n"
        "  • Una inferencia no respaldada por el contexto\n"
        "  • Una respuesta de 'conocimiento general' cuando debería usar los docs\n\n"
        "[bold]Solución:[/] después de generar, un LLM judge verifica\n"
        "si cada afirmación está en el contexto. Si no → regenerar.",
        border_style="blue",
        title="Hallucination Check",
    ))

    grafo = construir_grafo()

    console.print(Panel(
        grafo.get_graph().draw_ascii(),
        title="Estructura del grafo (con posible loop)",
        border_style="yellow",
    ))

    pregunta = "¿De qué trata el documento?"
    console.print(f"\n[bold]Pregunta:[/] {pregunta}")
    console.print("[bold]Ejecutando...[/]\n")

    estado_inicial: EstadoHallucination = {
        "pregunta": pregunta,
        "documentos": [],
        "contexto": "",
        "respuesta": "",
        "es_fiel": False,
        "razon_fidelidad": "",
        "regeneraciones": 0,
        "logs": [],
    }

    resultado = grafo.invoke(estado_inicial)

    color = "green" if resultado["es_fiel"] else "yellow"
    console.print(Panel(
        f"[bold]Fiel al contexto:[/] {resultado['es_fiel']}\n"
        f"[bold]Razón:[/] {resultado['razon_fidelidad']}\n"
        f"[bold]Regeneraciones:[/] {resultado['regeneraciones']}\n\n"
        f"[bold]Respuesta:[/]\n{resultado['respuesta'][:300]}...",
        title="Resultado final",
        border_style=color,
    ))

    console.print(Panel(
        "\n".join(f"  • {log}" for log in resultado["logs"]),
        title="Logs del flujo",
        border_style="dim",
    ))


if __name__ == "__main__":
    main()
