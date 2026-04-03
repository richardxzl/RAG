"""
08_batch.py — Batch: procesar múltiples preguntas en paralelo

Demuestra cómo .batch() ejecuta múltiples invocaciones concurrentemente,
en contraste con un loop de .invoke() que es puramente serial.

Tres escenarios:
  1. Batch básico: 5 preguntas, concurrencia sin límite
  2. max_concurrency: límite de 2 llamadas simultáneas (útil ante rate limits)
  3. Config por item: RunnableConfig individual para cada invocación
"""
import time
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()

# ── Pipeline sin cache (batch necesita llamar al LLM para cada pregunta) ──────
#
# build_query_chain() integra SemanticCache que cortocircuita el LLM.
# Para demostrar concurrencia real, construimos el pipeline directamente.
# .batch() paraleliza las llamadas al LLM; si el cache las absorbe,
# no hay nada que paralelizar y el demo pierde sentido.

def build_batch_pipeline():
    """
    Pipeline RAG directo, sin cache, apto para batch.

    La forma canónica con RunnablePassthrough.assign:
      1. assign recibe el dict {"question": ...}
      2. Agrega "context" invocando retriever + format_docs
      3. El dict enriquecido pasa al prompt, luego al LLM
    """
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


# ── Escenario 1: Batch básico ─────────────────────────────────────────────────

def escenario_batch_basico(pipeline, preguntas: list[str]) -> None:
    """
    .batch() envía todas las preguntas en paralelo.

    Por defecto usa un ThreadPoolExecutor con tantos workers como inputs.
    Cada invocación se ejecuta en un hilo separado, así que las llamadas
    al LLM se solapan en tiempo en lugar de ejecutarse una tras otra.
    """
    console.rule("[bold blue]Escenario 1 — Batch básico")
    console.print(
        "[dim]Todas las preguntas se envían a la vez. "
        "El tiempo total ≈ el tiempo de la pregunta más lenta, no la suma.[/]\n"
    )

    inputs = [{"question": q} for q in preguntas]

    # Medimos el tiempo del batch completo
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Procesando {len(preguntas)} preguntas en paralelo...", total=None)
        t0 = time.time()
        resultados = pipeline.batch(inputs)
        tiempo_batch = time.time() - t0
        progress.update(task, description="Completado")

    # Estimación serial: asumimos que el tiempo promedio por pregunta
    # es el tiempo_batch / N (no tenemos tiempos individuales en batch).
    # Usamos tiempo_batch como baseline y proyectamos el serial.
    tiempo_estimado_serial = tiempo_batch * len(preguntas)

    tabla = Table(show_header=True, header_style="bold magenta", show_lines=True)
    tabla.add_column("#", style="dim", width=3)
    tabla.add_column("Pregunta", style="cyan", max_width=35)
    tabla.add_column("Respuesta (80 chars)", max_width=82)

    for i, (pregunta, respuesta) in enumerate(zip(preguntas, resultados), 1):
        tabla.add_row(
            str(i),
            pregunta,
            respuesta[:80] + ("…" if len(respuesta) > 80 else ""),
        )

    console.print(tabla)
    console.print(f"\n[bold green]Tiempo real (batch):[/]   {tiempo_batch:.2f}s")
    console.print(f"[bold red]Tiempo estimado serial:[/] {tiempo_estimado_serial:.2f}s  "
                  f"[dim](= {tiempo_batch:.2f}s × {len(preguntas)} preguntas)[/]")
    console.print(
        f"[bold yellow]Ahorro aproximado:[/]     "
        f"{tiempo_estimado_serial - tiempo_batch:.2f}s "
        f"([dim]{((1 - tiempo_batch / tiempo_estimado_serial) * 100):.0f}% menos[/])\n"
    )


# ── Escenario 2: max_concurrency ──────────────────────────────────────────────

def escenario_max_concurrency(pipeline, preguntas: list[str]) -> None:
    """
    max_concurrency=2 limita a 2 llamadas simultáneas al LLM.

    ¿Por qué importa?
    - Las APIs de LLM tienen rate limits: N requests por minuto (RPM)
      y M tokens por minuto (TPM). Sin límite, un batch grande puede
      disparar HTTP 429 (Too Many Requests) y hacer fallar todo el lote.
    - Con max_concurrency controlamos la presión sobre la API.
    - Tradeoff: más lento que concurrencia total, pero más robusto.

    Ejemplo real: Anthropic Haiku en tier gratuito → 5 RPM.
    Con max_concurrency=2 y 5 preguntas: procesa 2, espera, procesa 2,
    espera, procesa 1. Nunca supera el límite.
    """
    console.rule("[bold blue]Escenario 2 — max_concurrency=2")
    console.print(
        "[dim]Máximo 2 llamadas simultáneas. "
        "Esencial para respetar rate limits de la API.[/]\n"
    )

    inputs = [{"question": q} for q in preguntas]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Procesando con concurrencia limitada...", total=None)
        t0 = time.time()
        # max_concurrency se pasa como parámetro directo de .batch()
        resultados = pipeline.batch(inputs, max_concurrency=2)
        tiempo_batch = time.time() - t0
        progress.update(task, description="Completado")

    tabla = Table(show_header=True, header_style="bold magenta", show_lines=True)
    tabla.add_column("#", style="dim", width=3)
    tabla.add_column("Pregunta", style="cyan", max_width=35)
    tabla.add_column("Respuesta (80 chars)", max_width=82)

    for i, (pregunta, respuesta) in enumerate(zip(preguntas, resultados), 1):
        tabla.add_row(
            str(i),
            pregunta,
            respuesta[:80] + ("…" if len(respuesta) > 80 else ""),
        )

    console.print(tabla)
    console.print(f"\n[bold green]Tiempo total (max_concurrency=2):[/] {tiempo_batch:.2f}s")
    console.print(
        "[dim]Comparar con Escenario 1: será más lento, "
        "pero no excede 2 requests simultáneos.[/]\n"
    )


# ── Escenario 3: Config por item ──────────────────────────────────────────────

def escenario_config_por_item(pipeline, preguntas: list[str]) -> None:
    """
    RunnableConfig permite pasar configuración individual a cada invocación.

    Casos de uso reales:
    - tags: identificar qué invocación es cuál en los traces de LangSmith
    - metadata: adjuntar info del usuario, del documento, del experimento
    - callbacks: handlers diferentes por item (e.g., logging a distintos destinos)
    - run_name: nombre descriptivo en el trace

    .batch() acepta una lista de configs, una por input.
    La lista debe tener exactamente la misma longitud que la lista de inputs.
    """
    console.rule("[bold blue]Escenario 3 — RunnableConfig por item")
    console.print(
        "[dim]Cada invocación recibe su propia configuración: "
        "tags, metadata, run_name.[/]\n"
    )

    inputs = [{"question": q} for q in preguntas]

    # Una RunnableConfig por cada pregunta
    configs: list[RunnableConfig] = [
        RunnableConfig(
            tags=[f"pregunta-{i}", "batch-demo", "escenario-3"],
            metadata={
                "pregunta_index": i,
                "pregunta_texto": q,
                "experimento": "08_batch_config_por_item",
            },
            run_name=f"rag-batch-pregunta-{i}",
        )
        for i, q in enumerate(preguntas, 1)
    ]

    # Mostrar las configs antes de ejecutar
    tabla_configs = Table(show_header=True, header_style="bold cyan")
    tabla_configs.add_column("#", style="dim", width=3)
    tabla_configs.add_column("run_name")
    tabla_configs.add_column("tags")
    tabla_configs.add_column("metadata keys")

    for i, cfg in enumerate(configs, 1):
        tabla_configs.add_row(
            str(i),
            cfg.get("run_name", "—"),
            str(cfg.get("tags", [])),
            str(list(cfg.get("metadata", {}).keys())),
        )

    console.print("[bold]Configuraciones que se pasarán a cada invocación:[/]")
    console.print(tabla_configs)
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Procesando con config individual...", total=None)
        t0 = time.time()
        # .batch(inputs, config) — config es la lista de RunnableConfig
        resultados = pipeline.batch(inputs, configs)
        tiempo_batch = time.time() - t0
        progress.update(task, description="Completado")

    tabla = Table(show_header=True, header_style="bold magenta", show_lines=True)
    tabla.add_column("#", style="dim", width=3)
    tabla.add_column("run_name", style="dim cyan")
    tabla.add_column("Pregunta", style="cyan", max_width=30)
    tabla.add_column("Respuesta (80 chars)", max_width=82)

    for i, (cfg, pregunta, respuesta) in enumerate(zip(configs, preguntas, resultados), 1):
        tabla.add_row(
            str(i),
            cfg.get("run_name", "—"),
            pregunta,
            respuesta[:80] + ("…" if len(respuesta) > 80 else ""),
        )

    console.print("[bold]Resultados:[/]")
    console.print(tabla)
    console.print(
        f"\n[bold green]Tiempo total:[/] {tiempo_batch:.2f}s\n"
        "[dim]En LangSmith, cada invocación aparece con su run_name y tags.[/]\n"
    )


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_demo():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Batch (1.5)")
    console.print(
        Panel(
            "[bold]¿Qué demuestra este script?[/]\n\n"
            "• [cyan].batch()[/] ejecuta N invocaciones en paralelo usando threads\n"
            "• El tiempo total ≈ el de la invocación más lenta, no la suma de todas\n"
            "• [cyan]max_concurrency[/] evita saturar la API con demasiadas llamadas simultáneas\n"
            "• [cyan]RunnableConfig[/] por item permite tracing y metadata individual",
            border_style="blue",
        )
    )

    preguntas = [
        "¿De qué trata el documento principal?",
        "¿Cuáles son los conceptos más importantes mencionados?",
        "¿Qué metodología o enfoque se describe?",
        "¿Hay ejemplos concretos en el texto?",
        "¿Cuál es la conclusión o punto central del documento?",
    ]

    console.print(f"\n[bold]Preguntas a procesar:[/] {len(preguntas)}")
    for i, q in enumerate(preguntas, 1):
        console.print(f"  [dim]{i}.[/] {q}")
    console.print()

    pipeline = build_batch_pipeline()

    escenario_batch_basico(pipeline, preguntas)
    escenario_max_concurrency(pipeline, preguntas)
    escenario_config_por_item(pipeline, preguntas)

    # ── Tabla resumen final ──
    console.rule("[bold]Resumen: ¿Cuándo usar cada opción?")
    resumen = Table(show_header=True, header_style="bold magenta", show_lines=True)
    resumen.add_column("Opción", style="cyan")
    resumen.add_column("Cuándo usarla")
    resumen.add_column("Cuándo evitarla")

    resumen.add_row(
        ".batch(inputs)",
        "Evaluar un dataset, procesar preguntas en bulk, máxima velocidad",
        "Si el bottleneck es el retriever local (no ganas nada con threads)",
    )
    resumen.add_row(
        "max_concurrency=N",
        "APIs con rate limits, entornos con recursos limitados",
        "Si tienes cuota generosa y necesitas máxima velocidad",
    )
    resumen.add_row(
        "Config por item",
        "Tracing en LangSmith, metadata de experimento, callbacks distintos",
        "Scripts de uso único donde no necesitas observabilidad",
    )
    resumen.add_row(
        ".abatch() (async)",
        "Entornos async (FastAPI, notebooks), para no bloquear el event loop",
        "Scripts síncronos — .batch() con threads es suficiente",
    )

    console.print(resumen)


if __name__ == "__main__":
    run_demo()
