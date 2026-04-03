"""
32_latencia_por_componente.py — Módulo 6.4: Métricas de latencia por componente

Mide y visualiza el tiempo que consume cada parte del pipeline RAG:
  - Retriever (ChromaDB, embedding del query)
  - Format docs
  - Prompt rendering
  - LLM (el mayor cuello de botella)
  - Total end-to-end

Permite identificar dónde optimizar: cache, chunk size, modelo más rápido, etc.
"""
import time
import statistics
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag.retriever import get_retriever
from rag.chain import get_llm, QUERY_PROMPT, format_docs
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

console = Console()

QUERIES = [
    "¿De qué trata el documento?",
    "¿Qué tecnologías se mencionan?",
    "Resume los puntos principales",
]
N_RUNS = 2  # repeticiones por query para promediar


def medir_paso(nombre: str, fn, *args, **kwargs):
    """Ejecuta fn y retorna (resultado, duracion_ms)."""
    t0 = time.perf_counter()
    resultado = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    return resultado, ms


def benchmark_pipeline(query: str) -> dict[str, float]:
    """
    Ejecuta el pipeline RAG midiendo cada paso por separado.
    Retorna un dict con la duración en ms de cada componente.
    """
    retriever = get_retriever()
    llm = get_llm()

    # Paso 1: embed del query + búsqueda en ChromaDB
    docs, ms_retriever = medir_paso("retriever", retriever.invoke, query)

    # Paso 2: formatear docs
    context, ms_format = medir_paso("format_docs", format_docs, docs)

    # Paso 3: renderizar el prompt (crear los mensajes)
    prompt_value, ms_prompt = medir_paso(
        "prompt_render",
        QUERY_PROMPT.format_messages,
        context=context,
        question=query,
    )

    # Paso 4: LLM
    chain_llm = llm | StrOutputParser()
    answer, ms_llm = medir_paso("llm", chain_llm.invoke, prompt_value)

    return {
        "retriever_ms": ms_retriever,
        "format_ms": ms_format,
        "prompt_ms": ms_prompt,
        "llm_ms": ms_llm,
        "total_ms": ms_retriever + ms_format + ms_prompt + ms_llm,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 6.4: Latencia por Componente")

    resultados_por_query = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Midiendo latencias...", total=len(QUERIES) * N_RUNS)

        for query in QUERIES:
            runs = []
            for _ in range(N_RUNS):
                r = benchmark_pipeline(query)
                runs.append(r)
                progress.advance(task)
            resultados_por_query[query] = runs

    # Promediar por query
    promedios = {}
    for query, runs in resultados_por_query.items():
        promedios[query] = {
            key: statistics.mean(r[key] for r in runs)
            for key in runs[0]
        }

    # Tabla de latencias por componente
    table = Table(title="Latencia por componente (ms)", show_header=True, header_style="bold magenta")
    table.add_column("Query", max_width=35)
    table.add_column("Retriever", width=10)
    table.add_column("Format", width=8)
    table.add_column("Prompt", width=8)
    table.add_column("LLM", width=10)
    table.add_column("TOTAL", width=10)

    for query, avg in promedios.items():
        total = avg["total_ms"]
        def pct(ms): return f"{ms:.0f} ({ms/total*100:.0f}%)"
        table.add_row(
            query[:33] + "...",
            pct(avg["retriever_ms"]),
            f"{avg['format_ms']:.1f}",
            f"{avg['prompt_ms']:.1f}",
            pct(avg["llm_ms"]),
            f"[bold]{avg['total_ms']:.0f}[/]",
        )

    console.print(table)

    # Promedios globales
    global_avg = {
        key: statistics.mean(avg[key] for avg in promedios.values())
        for key in list(promedios.values())[0]
    }

    console.print("\n[bold]Distribución global de latencia:[/]")
    componentes = [
        ("LLM", "llm_ms"),
        ("Retriever", "retriever_ms"),
        ("Prompt render", "prompt_ms"),
        ("Format docs", "format_ms"),
    ]
    total_global = global_avg["total_ms"]
    for nombre, key in sorted(componentes, key=lambda x: -global_avg[x[1]]):
        ms = global_avg[key]
        pct = ms / total_global * 100
        bar = "█" * int(pct / 5)
        console.print(f"  {nombre:15} {ms:6.0f}ms  {bar} {pct:.0f}%")

    # Recomendaciones
    llm_pct = global_avg["llm_ms"] / total_global * 100
    ret_pct = global_avg["retriever_ms"] / total_global * 100

    recomendaciones = []
    if llm_pct > 80:
        recomendaciones.append("LLM domina (>80%) → considera semantic cache (módulo 3 cache) o modelo más rápido")
    if ret_pct > 20:
        recomendaciones.append("Retriever tarda mucho (>20%) → revisa chunk size o usa retrieval cache")

    if recomendaciones:
        console.print(Panel(
            "\n".join(f"· {r}" for r in recomendaciones),
            title="Recomendaciones de optimización",
            border_style="yellow",
        ))


if __name__ == "__main__":
    main()
