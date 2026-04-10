"""
30_callbacks.py — Módulo 6.2: Callbacks de LangChain

Los callbacks son hooks que LangChain llama en eventos del pipeline:
  on_chain_start / on_chain_end
  on_llm_start / on_llm_end / on_llm_new_token
  on_retriever_start / on_retriever_end
  on_tool_start / on_tool_end

Implementamos tres callbacks útiles:
  1. TimingCallback — mide duración por componente
  2. LoggingCallback — loguea cada evento con contexto
  3. MetricsCallback — acumula métricas para análisis
"""
import time
import logging
from collections import defaultdict
from typing import Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Callback 1: TimingCallback ────────────────────────────────────────────────

class TimingCallback(BaseCallbackHandler):
    """Mide el tiempo de cada componente del pipeline."""

    def __init__(self):
        self.timings: dict[str, list[float]] = defaultdict(list)
        self._starts: dict[str, float] = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._starts["llm"] = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs):
        if "llm" in self._starts:
            self.timings["llm"].append(time.time() - self._starts.pop("llm"))

    def on_retriever_start(self, serialized, query, **kwargs):
        self._starts["retriever"] = time.time()

    def on_retriever_end(self, documents, **kwargs):
        if "retriever" in self._starts:
            self.timings["retriever"].append(time.time() - self._starts.pop("retriever"))

    def on_chain_start(self, serialized, inputs, **kwargs):
        chain_name = (serialized or {}).get("name", "chain")
        self._starts[f"chain_{chain_name}"] = time.time()

    def on_chain_end(self, outputs, **kwargs):
        # Limpiar cualquier chain que haya terminado
        for key in list(self._starts.keys()):
            if key.startswith("chain_"):
                self.timings[key].append(time.time() - self._starts.pop(key))
                break

    def report(self) -> dict[str, float]:
        return {
            step: sum(times) * 1000 / len(times)
            for step, times in self.timings.items()
            if times
        }


# ── Callback 2: LoggingCallback ───────────────────────────────────────────────

class LoggingCallback(BaseCallbackHandler):
    """Loguea cada evento del pipeline con información relevante."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.events: list[dict] = []

    def _log(self, event: str, data: dict):
        entry = {"event": event, "timestamp": time.time(), **data}
        self.events.append(entry)
        if self.verbose:
            console.print(f"  [dim cyan][callback] {event}[/] — {data}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        model = serialized.get("kwargs", {}).get("model", "?")
        self._log("llm_start", {"model": model, "num_prompts": len(prompts)})

    def on_llm_end(self, response: LLMResult, **kwargs):
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("usage", {})
        self._log("llm_end", {"token_usage": token_usage})

    def on_llm_new_token(self, token: str, **kwargs):
        # Solo loguear el primer y último token para no spamear
        pass

    def on_retriever_start(self, serialized, query, **kwargs):
        self._log("retriever_start", {"query": query[:60]})

    def on_retriever_end(self, documents, **kwargs):
        self._log("retriever_end", {"num_docs": len(documents)})

    def on_chain_error(self, error: Exception, **kwargs):
        self._log("chain_error", {"error": str(error)})

    def on_llm_error(self, error: Exception, **kwargs):
        self._log("llm_error", {"error": str(error)})


# ── Callback 3: MetricsCallback ───────────────────────────────────────────────

class MetricsCallback(BaseCallbackHandler):
    """Acumula métricas a lo largo de múltiples invocaciones."""

    def __init__(self):
        self.total_llm_calls = 0
        self.total_retriever_calls = 0
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.errors = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.total_llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs):
        if response.llm_output:
            usage = response.llm_output.get("usage", {})
            self.total_tokens_input += usage.get("input_tokens", 0)
            self.total_tokens_output += usage.get("output_tokens", 0)

    def on_retriever_start(self, serialized, query, **kwargs):
        self.total_retriever_calls += 1

    def on_chain_error(self, error, **kwargs):
        self.errors += 1

    def on_llm_error(self, error, **kwargs):
        self.errors += 1

    def report(self) -> dict:
        return {
            "llm_calls": self.total_llm_calls,
            "retriever_calls": self.total_retriever_calls,
            "tokens_input": self.total_tokens_input,
            "tokens_output": self.total_tokens_output,
            "errors": self.errors,
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

def build_pipeline_con_callbacks(callbacks: list):
    retriever = get_retriever()
    llm = get_llm()

    def run(question: str) -> str:
        docs = retriever.invoke(question, config={"callbacks": callbacks})
        context = format_docs(docs)
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | QUERY_PROMPT
            | llm
            | StrOutputParser()
        )
        return chain.invoke({"question": question, "docs": docs}, config={"callbacks": callbacks})

    return run


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 6.2: Callbacks")

    timing_cb = TimingCallback()
    logging_cb = LoggingCallback(verbose=True)
    metrics_cb = MetricsCallback()
    callbacks = [timing_cb, logging_cb, metrics_cb]

    pipeline = build_pipeline_con_callbacks(callbacks)

    questions = ["¿De qué trata el documento?", "¿Qué tecnologías se mencionan?"]

    for q in questions:
        console.print(f"\n[bold]Query:[/] {q}")
        answer = pipeline(q)
        console.print(Panel(answer[:200], border_style="green"))

    # Reporte de timing
    timings = timing_cb.report()
    if timings:
        t = Table(title="Timing por componente", show_header=True, header_style="bold magenta")
        t.add_column("Componente")
        t.add_column("Duración promedio (ms)")
        for step, ms in sorted(timings.items(), key=lambda x: -x[1]):
            t.add_row(step, f"{ms:.1f}")
        console.print(t)

    # Reporte de métricas acumuladas
    metrics = metrics_cb.report()
    console.print(Panel(
        "\n".join(f"  {k}: {v}" for k, v in metrics.items()),
        title="Métricas acumuladas",
        border_style="cyan",
    ))

    # Total de eventos logueados
    console.print(f"\n  [dim]{len(logging_cb.events)} eventos capturados por LoggingCallback[/]")


if __name__ == "__main__":
    main()
