"""
07_streaming.py — Streaming: respuestas token por token en tiempo real

Demuestra tres formas de hacer streaming sobre un pipeline RAG en LangChain:

  1. Streaming simple con print()   — la forma más directa
  2. Streaming con Rich Live        — UX en consola, tokens en pantalla en tiempo real
  3. Streaming con stream_events()  — visibilidad de bajo nivel del pipeline
"""
import sys
import time
import logging
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Builder del pipeline RAG para streaming ─────────────────────────────────
#
# IMPORTANTE: No usamos build_query_chain() del módulo porque ese wrapper
# tiene un SemanticCache integrado. El cache intercepta la respuesta completa
# y la retorna de una vez — nunca llega al LLM, por lo que no hay nada
# que streamear. Para streaming necesitamos el pipeline "desnudo".

def build_streaming_pipeline():
    """
    Pipeline RAG minimal sin cache.
    La cadena acepta {"question": str, "docs": list[Document]}.
    Retorna un Runnable que SÍ puede streamear porque no hay capa de cache.
    """
    llm = get_llm()

    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["docs"]),
        )
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )


def retrieve_docs(question: str):
    """Recupera los documentos relevantes. El retriever NO streamea."""
    retriever = get_retriever()
    return retriever.invoke(question)


# ── Variante 1: Streaming simple con print ───────────────────────────────────

def demo_streaming_simple(question: str) -> None:
    """
    La forma más directa de streaming: iterar sobre .stream() e imprimir
    cada chunk a medida que llega.

    .stream() retorna un generador. Cada elemento es un string (porque
    StrOutputParser convierte AIMessageChunk.content → str).
    Sin StrOutputParser, cada elemento sería un AIMessageChunk.
    """
    console.rule("[bold blue]Variante 1 — Streaming simple con print()")
    console.print(f"\n[bold]Pregunta:[/] {question}")
    console.print("[dim]Recuperando documentos...[/]")

    docs = retrieve_docs(question)
    console.print(f"[dim]  → {len(docs)} documentos recuperados[/]\n")

    pipeline = build_streaming_pipeline()

    console.print("[bold green]Respuesta (tokens en tiempo real):[/]")
    console.print("─" * 60)

    token_count = 0
    t0 = time.time()

    for chunk in pipeline.stream({"question": question, "docs": docs}):
        # chunk es un string: puede ser una palabra, parte de una, o "\n"
        print(chunk, end="", flush=True)
        token_count += 1

    elapsed = round((time.time() - t0) * 1000)
    print()  # salto de línea final
    console.print("─" * 60)
    console.print(f"\n[dim]  {token_count} chunks en {elapsed}ms[/]")


# ── Variante 2: Streaming con Rich Live ──────────────────────────────────────

def demo_streaming_rich_live(question: str) -> None:
    """
    Usa rich.live.Live para re-renderizar el panel completo en cada chunk.

    La estrategia: acumular todos los chunks en un string, y en cada
    iteración redibujar el panel con el contenido acumulado hasta ese momento.
    El resultado visual es una respuesta que "crece" en pantalla.

    Nota: Markdown() re-parsea el string completo en cada actualización.
    Para respuestas largas, Text() es más eficiente.
    """
    console.rule("[bold blue]Variante 2 — Streaming con Rich Live")
    console.print(f"\n[bold]Pregunta:[/] {question}")
    console.print("[dim]Recuperando documentos...[/]")

    docs = retrieve_docs(question)
    console.print(f"[dim]  → {len(docs)} documentos recuperados[/]\n")

    pipeline = build_streaming_pipeline()

    accumulated = ""
    token_count = 0
    t0 = time.time()

    with Live(
        Panel("", title=f"[bold green]{question}[/]", border_style="green"),
        console=console,
        refresh_per_second=15,  # máximo 15 re-renders por segundo
    ) as live:
        for chunk in pipeline.stream({"question": question, "docs": docs}):
            accumulated += chunk
            token_count += 1

            # Actualizar el panel con el contenido acumulado
            live.update(
                Panel(
                    accumulated,
                    title=f"[bold green]{question}[/]",
                    border_style="green",
                    subtitle=f"[dim]{token_count} chunks...[/]",
                )
            )

    elapsed = round((time.time() - t0) * 1000)
    console.print(f"[dim]  Completado: {token_count} chunks en {elapsed}ms[/]")


# ── Variante 3: Streaming con stream_events ──────────────────────────────────

def demo_streaming_events(question: str) -> None:
    """
    .stream_events() expone el flujo de eventos interno del pipeline.
    Cada evento tiene: event, name, data, run_id, tags.

    Eventos relevantes:
      on_chain_start   → un Runnable empezó a ejecutarse
      on_chain_stream  → un Runnable emitió un chunk intermedio
      on_chain_end     → un Runnable terminó
      on_llm_start     → el LLM recibió su input
      on_llm_stream    → el LLM emitió un token
      on_llm_end       → el LLM terminó de generar
      on_retriever_start / on_retriever_end → cuando hay retriever en el pipe

    version="v2" es la API actual (v1 está deprecated).
    include_names filtra por nombre del componente para reducir el ruido.
    """
    console.rule("[bold blue]Variante 3 — Streaming con stream_events()")
    console.print(f"\n[bold]Pregunta:[/] {question}")
    console.print("[dim]Recuperando documentos...[/]")

    docs = retrieve_docs(question)
    console.print(f"[dim]  → {len(docs)} documentos recuperados[/]\n")

    pipeline = build_streaming_pipeline()

    # Tabla de eventos para mostrar "lo que pasa bajo el capó"
    console.print("[bold yellow]Eventos del pipeline:[/]")
    console.print("─" * 60)

    llm_response = ""
    event_count = 0

    for event in pipeline.stream_events(
        {"question": question, "docs": docs},
        version="v2",
    ):
        event_type = event["event"]
        event_name = event.get("name", "")
        event_count += 1

        if event_type == "on_chain_start":
            console.print(
                f"[dim cyan]  ▶ chain_start   [/][cyan]{event_name}[/]"
            )

        elif event_type == "on_chain_end":
            # Solo mostrar end de cadenas relevantes (no las intermedias de StrOutputParser)
            if event_name in ("RunnableSequence", "RunnablePassthrough", "ChatPromptTemplate"):
                console.print(
                    f"[dim cyan]  ■ chain_end     [/][cyan]{event_name}[/]"
                )

        elif event_type == "on_llm_start":
            console.print(
                f"[bold magenta]  ▶ llm_start     [/][magenta]{event_name}[/] "
                f"[dim]← aquí empieza el streaming real[/]"
            )

        elif event_type == "on_llm_stream":
            # Cada token del LLM — acumular para mostrar la respuesta al final
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                llm_response += chunk.content
                # Mostrar el token crudo (sin salto de línea) como indicador visual
                console.print(
                    f"[dim green]  ~ llm_stream    token=[/][green]{repr(chunk.content[:20])}[/]",
                    highlight=False,
                )

        elif event_type == "on_llm_end":
            console.print(
                f"[bold magenta]  ■ llm_end       [/][magenta]{event_name}[/]"
            )

        elif event_type == "on_retriever_start":
            console.print(
                f"[bold blue]  ▶ retriever_start [/][blue]{event_name}[/]"
            )

        elif event_type == "on_retriever_end":
            docs_in_event = event["data"].get("output", [])
            console.print(
                f"[bold blue]  ■ retriever_end   [/][blue]{event_name}[/] "
                f"[dim]{len(docs_in_event) if isinstance(docs_in_event, list) else '?'} docs[/]"
            )

    console.print("─" * 60)
    console.print(f"[dim]  Total eventos procesados: {event_count}[/]\n")

    # Mostrar la respuesta acumulada de los eventos llm_stream
    if llm_response:
        console.print(Panel(
            llm_response,
            title="[bold green]Respuesta reconstruida desde eventos llm_stream[/]",
            border_style="green",
        ))


# ── Demo principal ────────────────────────────────────────────────────────────

def run_demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Streaming (1.4)")

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "¿De qué trata el documento?"

    console.print(f"\n[bold]Pregunta para las tres demos:[/] {question}")
    console.print("[dim]El retriever se invoca una vez por variante (no hay cache).[/]\n")

    # ── Variante 1 ──
    demo_streaming_simple(question)
    console.print()

    # ── Variante 2 ──
    demo_streaming_rich_live(question)
    console.print()

    # ── Variante 3 ──
    demo_streaming_events(question)

    # ── Resumen comparativo ──
    console.rule("[bold]Resumen: ¿Cuándo usar cada variante?[/]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Variante", style="cyan")
    table.add_column("Método")
    table.add_column("Mejor para")

    table.add_row(
        "1 — Simple",
        ".stream() + print()",
        "Scripts rápidos, debugging, pipelines sin UI",
    )
    table.add_row(
        "2 — Rich Live",
        ".stream() + Live()",
        "CLIs con UX cuidada, demos, herramientas internas",
    )
    table.add_row(
        "3 — Eventos",
        ".stream_events()",
        "Observabilidad, debugging avanzado, logging estructurado",
    )

    console.print(table)
    console.print(
        "\n[dim]Tip: python 07_streaming.py '¿tu pregunta aquí?'[/]"
    )


if __name__ == "__main__":
    run_demo()
