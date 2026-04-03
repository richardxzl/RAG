"""
62_health_checks.py — Módulo 12.4: Health checks

Liveness y readiness probes para la API del RAG.
  /health → liveness:   ¿el proceso responde?   (Kubernetes: reiniciar si falla)
  /ready  → readiness:  ¿puede servir tráfico?  (Kubernetes: sacar del LB si falla)
Demo: python 62_health_checks.py
"""
import os
import time
import logging
from enum import Enum
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.retriever import get_vectorstore
from rag.config import REDIS_URL

console = Console()

# ── Modelos ───────────────────────────────────────────────────

class HealthStatus(str, Enum):
    ok = "ok"
    degraded = "degraded"
    down = "down"

class ComponentCheck(BaseModel):
    status: HealthStatus
    latency_ms: float
    detail: str = ""

class ReadinessResponse(BaseModel):
    status: HealthStatus
    checks: dict[str, ComponentCheck]
    uptime_seconds: float

# ── App ───────────────────────────────────────────────────────

app = FastAPI(title="RAG API")
_start_time = time.time()


@app.get("/health")
def health():
    """
    Liveness probe — debe ser TRIVIAL.
    Solo confirma que el proceso está vivo y acepta conexiones.
    No comprueba dependencias externas.
    """
    return {"status": "ok", "timestamp": time.time()}


@app.get("/ready", response_model=ReadinessResponse)
def ready():
    """
    Readiness probe — comprueba TODAS las dependencias críticas.
    Si alguna está down, el pod no debería recibir tráfico.
    """
    checks: dict[str, ComponentCheck] = {}
    overall = HealthStatus.ok

    # ── 1. ChromaDB ──────────────────────────────────────────
    t0 = time.time()
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        checks["chromadb"] = ComponentCheck(
            status=HealthStatus.ok,
            latency_ms=round((time.time() - t0) * 1000, 2),
            detail=f"{count} documentos indexados",
        )
    except Exception as e:
        checks["chromadb"] = ComponentCheck(
            status=HealthStatus.down,
            latency_ms=round((time.time() - t0) * 1000, 2),
            detail=str(e)[:100],
        )
        overall = HealthStatus.down

    # ── 2. Redis (cache — degraded si no está, no down) ──────
    t0 = time.time()
    try:
        import redis
        r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        checks["redis"] = ComponentCheck(
            status=HealthStatus.ok,
            latency_ms=round((time.time() - t0) * 1000, 2),
            detail="PONG",
        )
    except Exception as e:
        # Redis es opcional (cache). La API funciona sin él — degraded, no down.
        checks["redis"] = ComponentCheck(
            status=HealthStatus.degraded,
            latency_ms=round((time.time() - t0) * 1000, 2),
            detail=f"No disponible: {str(e)[:60]}",
        )
        if overall == HealthStatus.ok:
            overall = HealthStatus.degraded

    # ── 3. LLM — verificar que la API key existe ─────────────
    t0 = time.time()
    try:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY no configurada")
        checks["llm"] = ComponentCheck(
            status=HealthStatus.ok,
            latency_ms=round((time.time() - t0) * 1000, 2),
            detail="API key presente",
        )
    except Exception as e:
        checks["llm"] = ComponentCheck(
            status=HealthStatus.down,
            latency_ms=round((time.time() - t0) * 1000, 2),
            detail=str(e),
        )
        overall = HealthStatus.down

    return ReadinessResponse(
        status=overall,
        checks=checks,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


# ── Demo ──────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 12.4: Health Checks")

    client = TestClient(app)

    # Liveness
    console.print("\n[bold]GET /health (liveness)[/bold]")
    resp = client.get("/health")
    console.print(Panel(str(resp.json()), title=f"Status {resp.status_code}", border_style="green"))

    # Readiness
    console.print("\n[bold]GET /ready (readiness)[/bold]")
    resp = client.get("/ready")
    data = resp.json()

    status_colors = {"ok": "green", "degraded": "yellow", "down": "red"}
    overall_color = status_colors.get(data["status"], "white")

    table = Table(
        title=f"Readiness — Overall: [{overall_color}]{data['status'].upper()}[/{overall_color}]",
        show_lines=True,
    )
    table.add_column("Componente", style="cyan", width=12)
    table.add_column("Status", width=10)
    table.add_column("Latencia", width=10)
    table.add_column("Detalle", style="white")

    for name, check in data["checks"].items():
        color = status_colors.get(check["status"], "white")
        table.add_row(
            name,
            f"[{color}]{check['status'].upper()}[/{color}]",
            f"{check['latency_ms']} ms",
            check["detail"],
        )

    console.print(table)
    console.print(f"[dim]Uptime: {data['uptime_seconds']}s[/dim]")

    console.print(Panel(
        "Kubernetes:\n"
        "  livenessProbe:\n"
        "    httpGet: {path: /health, port: 8000}\n"
        "    periodSeconds: 5\n\n"
        "  readinessProbe:\n"
        "    httpGet: {path: /ready, port: 8000}\n"
        "    periodSeconds: 30\n"
        "    failureThreshold: 3\n\n"
        "Docker Compose:\n"
        "  healthcheck:\n"
        "    test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:8000/health\"]\n"
        "    interval: 30s\n"
        "    timeout: 10s\n"
        "    retries: 3",
        title="Configuración en orquestadores",
        border_style="yellow",
    ))


if __name__ == "__main__":
    main()
