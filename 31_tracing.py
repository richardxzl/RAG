"""
31_tracing.py — Módulo 6.3: Tracing con LangSmith o alternativa local

Demuestra dos opciones de tracing:
  1. LangSmith (nube): tracing visual completo con UI
  2. Alternativa local: tracing manual con OpenTelemetry-like structure

El tracing va más allá del logging: captura el árbol completo de llamadas,
inputs/outputs de cada nodo, y permite replay/debug de requests específicas.
"""
import os
import uuid
import time
import json
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel

console = Console()


# ── Opción 1: LangSmith ───────────────────────────────────────────────────────

def setup_langsmith():
    """
    Configura LangSmith si las credenciales están disponibles.
    Con estas variables de entorno activas, LangChain traza automáticamente.
    """
    langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not langsmith_key:
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "rag-learning"
    # LANGCHAIN_API_KEY ya está en el entorno

    console.print("[green]LangSmith configurado — todas las ejecuciones se trazarán[/]")
    console.print("[dim]Ver traces en: https://smith.langchain.com[/]")
    return True


# ── Opción 2: Tracer local (sin dependencias externas) ───────────────────────

@dataclass
class Span:
    """Representa un span de ejecución en el árbol de trazas."""
    trace_id: str
    span_id: str
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    error: Optional[str] = None
    children: list["Span"] = field(default_factory=list)
    parent_id: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def finish(self, outputs: dict = None, error: str = None):
        self.end_time = time.time()
        if outputs:
            self.outputs = outputs
        if error:
            self.error = error


class LocalTracer:
    """Tracer local que mantiene el árbol de spans en memoria."""

    def __init__(self):
        self._spans: dict[str, Span] = {}
        self._root_spans: list[Span] = []
        self._active_span: Optional[Span] = None

    def start_span(self, name: str, inputs: dict = None) -> Span:
        trace_id = self._active_span.trace_id if self._active_span else str(uuid.uuid4())[:8]
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:8],
            name=name,
            inputs=inputs or {},
            parent_id=self._active_span.span_id if self._active_span else None,
        )
        self._spans[span.span_id] = span

        if self._active_span:
            self._active_span.children.append(span)
        else:
            self._root_spans.append(span)

        self._active_span = span
        return span

    def end_span(self, span: Span, outputs: dict = None, error: str = None):
        span.finish(outputs=outputs, error=error)
        # Restaurar el parent como span activo
        if span.parent_id and span.parent_id in self._spans:
            self._active_span = self._spans[span.parent_id]
        else:
            self._active_span = None

    def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self._spans.values() if s.trace_id == trace_id]

    def render_tree(self, span: Span, tree: Tree = None) -> Tree:
        """Renderiza el árbol de spans con rich.Tree."""
        status = "[red]ERROR[/]" if span.error else f"[green]{span.duration_ms:.0f}ms[/]"
        label = f"[cyan]{span.name}[/] — {status}"

        if tree is None:
            node = Tree(label)
        else:
            node = tree.add(label)

        # Inputs/outputs como summary
        if span.inputs:
            summary = str(span.inputs)[:60]
            node.add(f"[dim]↳ input: {summary}[/]")
        if span.outputs:
            summary = str(span.outputs)[:60]
            node.add(f"[dim]↳ output: {summary}[/]")
        if span.error:
            node.add(f"[red]↳ error: {span.error}[/]")

        for child in span.children:
            self.render_tree(child, node)

        return node


# ── Pipeline RAG con tracing local ───────────────────────────────────────────

def query_con_tracing(question: str, tracer: LocalTracer) -> str:
    from rag.retriever import get_retriever
    from rag.chain import get_llm, QUERY_PROMPT, format_docs
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # Span raíz: toda la request
    root_span = tracer.start_span("rag_query", inputs={"question": question})

    try:
        # Span: retriever
        retriever_span = tracer.start_span("retriever", inputs={"query": question})
        retriever = get_retriever()
        docs = retriever.invoke(question)
        tracer.end_span(retriever_span, outputs={"num_docs": len(docs)})

        # Span: format_docs
        fmt_span = tracer.start_span("format_docs", inputs={"num_docs": len(docs)})
        context = format_docs(docs)
        tracer.end_span(fmt_span, outputs={"context_chars": len(context)})

        # Span: llm
        llm_span = tracer.start_span("llm", inputs={"context_chars": len(context)})
        llm = get_llm()
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | QUERY_PROMPT
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke({"question": question, "docs": docs})
        tracer.end_span(llm_span, outputs={"answer_chars": len(answer)})

        tracer.end_span(root_span, outputs={"success": True})
        return answer

    except Exception as e:
        tracer.end_span(root_span, error=str(e))
        raise


def main():
    console.rule("[bold blue]RAG Lab — Módulo 6.3: Tracing")

    # Intentar LangSmith primero
    langsmith_activo = setup_langsmith()

    if not langsmith_activo:
        console.print(Panel(
            "[yellow]LangSmith no configurado[/] (LANGCHAIN_API_KEY no encontrada)\n\n"
            "Para activarlo:\n"
            "  export LANGCHAIN_API_KEY='tu-key'\n"
            "  export LANGCHAIN_TRACING_V2='true'\n\n"
            "Usando tracer local en su lugar.",
            border_style="yellow",
        ))

    # Tracer local
    tracer = LocalTracer()

    console.print("\n[bold yellow]Ejecutando con tracer local:[/]\n")
    answer = query_con_tracing("¿De qué trata el documento?", tracer)

    # Renderizar el árbol de spans
    if tracer._root_spans:
        root = tracer._root_spans[-1]
        console.print("\n[bold]Árbol de trazas:[/]")
        tree = tracer.render_tree(root)
        console.print(tree)

        # Stats
        total_ms = root.duration_ms
        console.print(f"\n[dim]Total: {total_ms:.0f}ms[/]")
        for child in root.children:
            pct = (child.duration_ms / total_ms * 100) if total_ms > 0 else 0
            console.print(f"  {child.name}: {child.duration_ms:.0f}ms ({pct:.0f}%)")

    console.print(Panel(answer[:200], title="Respuesta", border_style="green"))


if __name__ == "__main__":
    main()
