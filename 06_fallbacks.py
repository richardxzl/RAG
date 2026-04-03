"""
06_fallbacks.py — Fallbacks: retry y modelo alternativo cuando falla un paso

Demuestra tres mecanismos de resiliencia en LCEL:

  A. .with_retry()        — mismo Runnable, reintentar N veces ante fallo transitorio
  B. .with_fallbacks()    — Runnable alternativo cuando el primario falla permanentemente
  C. Fallback RAG → LLM   — pipeline completo: si falla el retriever, responde sin contexto
"""
import logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.rule import Rule

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()

PREGUNTA_DEMO = "¿De qué trata el documento?"


# ── Caso A: .with_retry() ────────────────────────────────────────────────────

def make_flaky_step(fail_times: int):
    """
    Factory que crea un RunnableLambda que falla las primeras `fail_times`
    invocaciones y luego tiene éxito.

    Usa un closure con lista mutable para mantener el contador de intentos
    sin necesidad de una clase. La lista actúa como referencia compartida.
    """
    attempts = [0]  # lista mutable para que el closure pueda modificarla

    def flaky(question: str) -> str:
        attempts[0] += 1
        intento_actual = attempts[0]

        if intento_actual <= fail_times:
            console.print(
                f"   [bold red]✗ Intento {intento_actual}/{fail_times + 1} — "
                f"simulando fallo transitorio[/]"
            )
            raise Exception(f"Simulated API error (intento {intento_actual})")

        console.print(
            f"   [bold green]✓ Intento {intento_actual}/{fail_times + 1} — éxito[/]"
        )
        return question  # pasa la pregunta al siguiente paso

    return flaky


def demo_retry():
    """
    Caso A: Demuestra .with_retry().

    Construye un paso que falla 2 veces antes de tener éxito.
    .with_retry(stop_after_attempt=3) reintenta automáticamente hasta 3 veces.
    """
    console.print(Rule("[bold yellow]Caso A — .with_retry()[/]"))
    console.print(
        "[dim]Un paso falla las primeras 2 veces. "
        ".with_retry(stop_after_attempt=3) lo reintenta automáticamente.[/]\n"
    )

    llm = get_llm()

    # El paso inestable: falla 2 veces, tiene éxito en el 3er intento
    flaky_step = RunnableLambda(make_flaky_step(fail_times=2)).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=False,  # sin espera entre intentos para la demo
    )

    # Pipeline completo: paso inestable → retriever → LLM
    retriever = get_retriever()

    pipeline = (
        flaky_step
        | RunnablePassthrough.assign(
            question=lambda x: x,
            context=lambda x: format_docs(retriever.invoke(x)),
        )
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )

    console.print(f"[bold]Pregunta:[/] {PREGUNTA_DEMO}\n")

    try:
        respuesta = pipeline.invoke(PREGUNTA_DEMO)
        console.print(Panel(
            Markdown(respuesta),
            title="[green]Respuesta obtenida (tras reintentos)[/]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Error tras agotar reintentos:[/] {e}")

    console.print()


# ── Caso B: .with_fallbacks() ────────────────────────────────────────────────

def demo_fallbacks():
    """
    Caso B: Demuestra .with_fallbacks().

    El LLM primario se simula con un RunnableLambda que siempre lanza excepción.
    El fallback es el LLM real. LangChain prueba el primario primero; si falla,
    pasa al siguiente en la lista.
    """
    console.print(Rule("[bold yellow]Caso B — .with_fallbacks()[/]"))
    console.print(
        "[dim]LLM primario siempre falla (simulado). "
        ".with_fallbacks() cae automáticamente al LLM real.[/]\n"
    )

    llm_real = get_llm()
    retriever = get_retriever()

    # LLM primario simulado que siempre falla
    def llm_primario_roto(_input):
        console.print("   [bold red]✗ LLM primario falló (simulado)[/]")
        raise Exception("Simulated API error — modelo primario no disponible")

    llm_primario = RunnableLambda(llm_primario_roto)

    # Configurar fallback: primario → llm_real si primario falla
    llm_con_fallback = llm_primario.with_fallbacks(
        fallbacks=[llm_real],
        exceptions_to_handle=(Exception,),
    )

    pipeline = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
        )
        | QUERY_PROMPT
        | llm_con_fallback
        | StrOutputParser()
    )

    console.print(f"[bold]Pregunta:[/] {PREGUNTA_DEMO}\n")

    try:
        respuesta = pipeline.invoke({"question": PREGUNTA_DEMO})
        console.print(Panel(
            Markdown(respuesta),
            title="[green]Respuesta del fallback (LLM real)[/]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Error tras agotar fallbacks:[/] {e}")

    console.print()


# ── Caso C: Fallback completo del pipeline RAG ───────────────────────────────

def demo_rag_fallback():
    """
    Caso C: Fallback a nivel de pipeline completo.

    El pipeline RAG primario simula un fallo en el retriever.
    El fallback es un chain sin RAG que responde solo con el LLM,
    sin contexto de documentos.

    Esto modela un escenario real: ChromaDB caído o Redis caído.
    """
    console.print(Rule("[bold yellow]Caso C — Fallback completo del pipeline RAG[/]"))
    console.print(
        "[dim]El retriever falla (simulado). "
        "El pipeline cae a un chain LLM directo sin contexto RAG.[/]\n"
    )

    llm = get_llm()

    # ── Pipeline primario: RAG completo con retriever roto ──
    def retriever_roto(_question: str):
        console.print("   [bold red]✗ Retriever falló — ChromaDB no disponible (simulado)[/]")
        raise Exception("Simulated ChromaDB connection error")

    retriever_roto_runnable = RunnableLambda(retriever_roto)

    pipeline_rag = (
        retriever_roto_runnable  # simula que el retriever falla
        | RunnablePassthrough.assign(
            question=lambda x: x,
            context=lambda x: format_docs(x),
        )
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )

    # ── Pipeline fallback: LLM directo sin RAG ──
    from langchain_core.prompts import ChatPromptTemplate

    prompt_sin_rag = ChatPromptTemplate.from_template(
        """Eres un asistente útil. Responde la siguiente pregunta lo mejor que puedas.
No tienes acceso a documentos adicionales en este momento.

Pregunta: {question}

Respuesta (en español, clara y concisa):"""
    )

    def extraer_pregunta(question):
        """Adapta la entrada para el prompt sin RAG."""
        console.print(
            "   [bold yellow]⚡ Activando fallback: respondiendo sin contexto RAG[/]"
        )
        return {"question": question}

    pipeline_fallback = (
        RunnableLambda(extraer_pregunta)
        | prompt_sin_rag
        | llm
        | StrOutputParser()
    )

    # ── Componer con fallback ──
    pipeline_resiliente = pipeline_rag.with_fallbacks(
        fallbacks=[pipeline_fallback],
        exceptions_to_handle=(Exception,),
    )

    console.print(f"[bold]Pregunta:[/] {PREGUNTA_DEMO}\n")

    try:
        respuesta = pipeline_resiliente.invoke(PREGUNTA_DEMO)
        console.print(Panel(
            Markdown(respuesta),
            title="[yellow]Respuesta del fallback (LLM sin RAG)[/]",
            border_style="yellow",
            subtitle="[dim]Calidad reducida: sin contexto de documentos[/]",
        ))
    except Exception as e:
        console.print(f"[bold red]Error tras agotar todos los fallbacks:[/] {e}")

    console.print()


# ── Tabla resumen ─────────────────────────────────────────────────────────────

def mostrar_resumen():
    """Muestra tabla comparativa de los mecanismos de resiliencia."""
    console.print(Rule("[bold blue]Resumen: retry vs fallback[/]"))

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Mecanismo", style="cyan", min_width=18)
    table.add_column("¿Cuándo usarlo?", min_width=30)
    table.add_column("Tradeoff principal")

    table.add_row(
        ".with_retry()",
        "Errores transitorios: rate limits,\ntimeouts de red, picos de carga",
        "Aumenta la latencia por los reintentos;\nno sirve para errores permanentes",
    )
    table.add_row(
        ".with_fallbacks()",
        "Modelo no disponible, servicio caído,\nerror permanente del componente",
        "El fallback puede dar respuesta de\nmenor calidad (modelo más simple o sin RAG)",
    )
    table.add_row(
        "retry + fallback\ncombinados",
        "Primero reintentar N veces;\nsi sigue fallando, cambiar componente",
        "Mayor complejidad; diseñar con cuidado\nel orden de degradación",
    )

    console.print(table)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_demo():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Fallbacks (1.3)[/]")
    console.print(
        "\n[dim]Cada caso activa el mecanismo de resiliencia de forma visible. "
        "Observa los mensajes de fallo y recuperación.[/]\n"
    )

    demo_retry()
    demo_fallbacks()
    demo_rag_fallback()
    mostrar_resumen()


if __name__ == "__main__":
    run_demo()
