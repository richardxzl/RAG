"""
29_logging_estructurado.py — Módulo 6.1: Logging estructurado por paso

Implementa logging estructurado (JSON) para cada paso del pipeline RAG.
Permite correlacionar logs, medir tiempos por componente, y debugear en producción.

Estructura del log:
  - timestamp, level, message
  - request_id (correlaciona todos los logs de una request)
  - step (retriever, llm, cache, etc.)
  - duration_ms
  - metadata relevante del paso
"""
import json
import uuid
import time
import logging
import sys
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


# ── Logger estructurado ───────────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """Formatea los logs como JSON para facilitar el parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Campos extra pasados con extra={...}
        for key in ["request_id", "step", "duration_ms", "query", "num_docs",
                    "cache_hit", "model", "tokens", "error"]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logger(name: str = "rag", level: int = logging.DEBUG) -> logging.Logger:
    """Configura el logger estructurado."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# ── Context manager para medir duración ──────────────────────────────────────

class StepTimer:
    """Mide el tiempo de un paso y loguea el resultado."""

    def __init__(self, logger: logging.Logger, step: str, request_id: str, **extra):
        self.logger = logger
        self.step = step
        self.request_id = request_id
        self.extra = extra
        self.start = None

    def __enter__(self):
        self.start = time.time()
        self.logger.debug(
            f"Iniciando paso: {self.step}",
            extra={"request_id": self.request_id, "step": self.step, **self.extra},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = round((time.time() - self.start) * 1000)
        if exc_type:
            self.logger.error(
                f"Error en paso: {self.step}",
                extra={
                    "request_id": self.request_id,
                    "step": self.step,
                    "duration_ms": duration_ms,
                    "error": str(exc_val),
                },
            )
        else:
            self.logger.info(
                f"Paso completado: {self.step}",
                extra={
                    "request_id": self.request_id,
                    "step": self.step,
                    "duration_ms": duration_ms,
                    **self.extra,
                },
            )
        return False  # no suprimir la excepción


# ── Pipeline RAG con logging estructurado ────────────────────────────────────

def query_con_logging(question: str, logger: logging.Logger) -> str:
    """Ejecuta el pipeline RAG logueando cada paso con timing."""
    from rag.retriever import get_retriever
    from rag.chain import get_llm, QUERY_PROMPT, format_docs
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    request_id = str(uuid.uuid4())[:8]

    logger.info(
        "Request iniciada",
        extra={"request_id": request_id, "step": "request", "query": question[:80]},
    )

    # Paso 1: Retrieval
    docs = []
    with StepTimer(logger, "retriever", request_id, query=question[:80]):
        retriever = get_retriever()
        docs = retriever.invoke(question)

    logger.info(
        "Docs recuperados",
        extra={
            "request_id": request_id,
            "step": "retriever",
            "num_docs": len(docs),
            "sources": [d.metadata.get("source", "?") for d in docs],
        },
    )

    # Paso 2: Format context
    context = ""
    with StepTimer(logger, "format_docs", request_id):
        context = format_docs(docs)

    logger.debug(
        "Contexto formateado",
        extra={
            "request_id": request_id,
            "step": "format_docs",
            "context_chars": len(context),
        },
    )

    # Paso 3: LLM
    answer = ""
    with StepTimer(logger, "llm", request_id, model="claude-haiku"):
        llm = get_llm()
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | QUERY_PROMPT
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke({"question": question, "docs": docs})

    logger.info(
        "Request completada",
        extra={
            "request_id": request_id,
            "step": "request",
            "answer_chars": len(answer),
        },
    )

    return answer


def main():
    console.rule("[bold blue]RAG Lab — Módulo 6.1: Logging Estructurado")

    console.print(Panel(
        "[bold]Por qué logging estructurado (JSON) en vez de print():[/]\n\n"
        "· Parseable por herramientas (Datadog, Loki, CloudWatch)\n"
        "· request_id correlaciona todos los logs de una request\n"
        "· duration_ms permite detectar cuellos de botella\n"
        "· Niveles (DEBUG, INFO, ERROR) filtran por verbosidad\n\n"
        "[dim]En producción: los logs van a un agregador (ELK, Datadog).\n"
        "Puedes hacer queries como: 'todos los requests con duration_ms > 3000'[/]",
        border_style="blue",
    ))

    logger = setup_logger("rag.pipeline")

    console.print("\n[bold yellow]Ejecutando pipeline con logging:[/]\n")
    answer = query_con_logging("¿De qué trata el documento?", logger)

    console.print(Panel(answer[:300], title="Respuesta", border_style="green"))

    # Mostrar cómo se ve un log entry
    ejemplo_log = {
        "timestamp": "2026-03-30T10:00:01.234+00:00",
        "level": "INFO",
        "message": "Paso completado: retriever",
        "logger": "rag.pipeline",
        "request_id": "a3b7c9d2",
        "step": "retriever",
        "duration_ms": 145,
        "num_docs": 4,
        "sources": ["manual.pdf", "manual.pdf", "guia.md", "guia.md"],
    }
    console.print("\n[bold]Ejemplo de log entry (JSON):[/]")
    console.print(Syntax(
        json.dumps(ejemplo_log, indent=2, ensure_ascii=False),
        "json",
        theme="monokai",
    ))


if __name__ == "__main__":
    main()
