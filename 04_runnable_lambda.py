"""
04_runnable_lambda.py — RunnableLambda: funciones custom en el pipeline

Demuestra cómo convertir funciones Python en pasos de un pipeline LCEL
usando RunnableLambda. Incluye tres casos de uso reales:

  1. Normalización del input (pregunta limpia → mejor hit en cache)
  2. Logging de cada paso (debugear el pipeline sin romperlo)
  3. Enriquecimiento del output (agregar metadata a la respuesta)
"""
import time
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Funciones custom que convertiremos en Runnables ─────────────────────────

def normalize_question(question: str) -> str:
    """
    Normaliza la pregunta antes de buscar en el retriever.
    - Elimina espacios extra
    - Convierte a minúsculas
    - Elimina signos de puntuación al final

    ¿Por qué importa? El cache semántico es sensible a variaciones mínimas.
    "¿De qué trata?" y "de que trata?" deberían ser el mismo hit de cache.
    """
    normalized = question.strip().lower().rstrip("?!.,")
    if normalized != question:
        console.print(f"   [dim]normalize: '{question}' → '{normalized}'[/]")
    return normalized


def make_logger(step_name: str):
    """
    Factory que crea una función de logging para un paso específico.
    Retorna una función que loguea y pasa el valor sin modificarlo.

    Patrón: el RunnableLambda de logging es siempre un pass-through.
    """
    def log(value):
        tipo = type(value).__name__
        if isinstance(value, str):
            preview = value[:60].replace("\n", " ")
            console.print(f"   [dim cyan][{step_name}] {tipo}: \"{preview}...\"[/]")
        elif isinstance(value, list):
            console.print(f"   [dim cyan][{step_name}] {tipo}: {len(value)} elementos[/]")
        elif isinstance(value, dict):
            keys = list(value.keys())
            console.print(f"   [dim cyan][{step_name}] {tipo}: keys={keys}[/]")
        else:
            console.print(f"   [dim cyan][{step_name}] {tipo}[/]")
        return value  # CRÍTICO: siempre retornar el valor sin modificar
    return log


def add_timing(start_time: float):
    """
    Factory que crea una función que agrega el tiempo de ejecución al output.
    """
    def enrich(answer: str) -> dict:
        elapsed = time.time() - start_time
        return {
            "answer": answer,
            "elapsed_ms": round(elapsed * 1000),
        }
    return enrich


# ── Pipeline 1: Básico (sin RunnableLambda) ──────────────────────────────────

def build_basic_pipeline():
    """Pipeline RAG sin RunnableLambda — como punto de comparación."""
    retriever = get_retriever()
    llm = get_llm()

    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
        )
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )


# ── Pipeline 2: Con RunnableLambda ───────────────────────────────────────────

def build_instrumented_pipeline():
    """
    Pipeline RAG con RunnableLambda en tres puntos clave:
      - Antes del retriever: normalizar la pregunta
      - Después del retriever: loguear los docs recuperados
      - Al final: enriquecer el output con metadata de timing
    """
    retriever = get_retriever()
    llm = get_llm()

    # Cada función Python se convierte en un Runnable con RunnableLambda
    normalize   = RunnableLambda(normalize_question)
    log_docs    = RunnableLambda(make_logger("retriever"))
    log_context = RunnableLambda(make_logger("context"))

    # Pipeline con pasos custom insertados
    start_time = time.time()

    retriever_with_log = retriever | RunnableLambda(make_logger("retriever"))

    pipeline = (
        normalize                       # 1. Normalizar la pregunta
        | RunnablePassthrough.assign(
            question=lambda x: x,       # la pregunta normalizada queda en "question"
            context=lambda x: (
                log_context(            # 3. Loguear el contexto formateado
                    format_docs(
                        log_docs(       # 2. Loguear los docs crudos
                            retriever.invoke(x)
                        )
                    )
                )
            ),
        )
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
        | RunnableLambda(add_timing(start_time))  # 4. Enriquecer con timing
    )

    return pipeline


# ── Demo ─────────────────────────────────────────────────────────────────────

def run_demo():
    console.rule("[bold blue]RAG Lab — RunnableLambda (1.1)")

    questions = [
        "¿De qué trata el documento?",
        "de que trata el documento?",   # misma pregunta, forma diferente
    ]

    # ── Pipeline básico ──
    console.print("\n[bold yellow]Pipeline básico (sin RunnableLambda)[/]")
    basic = build_basic_pipeline()

    t0 = time.time()
    answer = basic.invoke({"question": questions[0]})
    elapsed = round((time.time() - t0) * 1000)
    console.print(Panel(Markdown(answer), title=f"Respuesta ({elapsed}ms)", border_style="dim"))

    # ── Pipeline con RunnableLambda ──
    console.print("\n[bold yellow]Pipeline con RunnableLambda (logging + normalización + timing)[/]")
    console.print("[dim]Observa los logs de cada paso:[/]\n")

    instrumented = build_instrumented_pipeline()

    for q in questions:
        console.print(f"\n[bold]Pregunta:[/] {q}")
        result = instrumented.invoke(q)

        console.print(Panel(
            Markdown(result["answer"]),
            title=f"Respuesta ({result['elapsed_ms']}ms)",
            border_style="green",
        ))

    # ── Tabla comparativa ──
    console.print("\n[bold]Resumen: ¿Cuándo usar RunnableLambda?[/]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Caso de uso", style="cyan")
    table.add_column("Ejemplo en este demo")
    table.add_column("Alternativa sin RunnableLambda")

    table.add_row(
        "Normalización de input",
        "normalize_question()",
        "Hacerlo fuera del pipeline (rompe la composabilidad)",
    )
    table.add_row(
        "Logging / debugging",
        "make_logger('paso')",
        "print() sueltos — no respetan el flujo del pipe",
    )
    table.add_row(
        "Enriquecimiento de output",
        "add_timing(start_time)",
        "Wrappear toda la invocación en try/except externo",
    )
    table.add_row(
        "Integrar código externo",
        "cualquier función Python",
        "No es posible directo con LCEL sin RunnableLambda",
    )

    console.print(table)


if __name__ == "__main__":
    run_demo()
