"""
61_rate_limiting.py — Módulo 12.3: Rate limiting

Limitar requests a la API para controlar costos del LLM y evitar abuso.
Dos enfoques: slowapi (decorador) y middleware manual (sin dependencias extra).
Demo: python 61_rate_limiting.py
"""
from collections import defaultdict
from time import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class QueryRequest(BaseModel):
    question: str


# ── Opción A: slowapi (pip install slowapi) ───────────────────

def build_app_slowapi() -> FastAPI:
    """Rate limiting con slowapi — el estándar para FastAPI."""
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded
    except ImportError:
        return None

    limiter = Limiter(key_func=get_remote_address)
    app = FastAPI(title="RAG API — slowapi")
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.post("/query")
    @limiter.limit("10/minute")  # 10 requests/minuto por IP
    def query(req: QueryRequest, request: Request):
        return {"answer": f"Respuesta a: {req.question}", "tier": "free"}

    @app.post("/query/premium")
    @limiter.limit("100/minute")  # tier premium — límite más alto
    def query_premium(req: QueryRequest, request: Request):
        return {"answer": f"Respuesta a: {req.question}", "tier": "premium"}

    return app


# ── Opción B: middleware manual (sin dependencias extra) ──────

def build_app_manual(limit: int = 10, window: float = 60.0) -> FastAPI:
    """
    Ventana deslizante en memoria.
    Sin dependencias extra, pero no persiste entre reinicios.
    Para producción: usar Redis como backend.
    """
    app = FastAPI(title="RAG API — Rate Limiting Manual")

    # {ip: [timestamps de los últimos N requests]}
    _requests: dict[str, list[float]] = defaultdict(list)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        ip = request.client.host if request.client else "testclient"
        now = time()

        # Limpiar timestamps fuera de la ventana
        _requests[ip] = [t for t in _requests[ip] if now - t < window]

        if len(_requests[ip]) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "limit": limit,
                    "window_seconds": window,
                },
                headers={"Retry-After": str(int(window))},
            )

        _requests[ip].append(now)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/query")
    def query(req: QueryRequest):
        return {"answer": f"Respuesta a: {req.question}"}

    return app


def main():
    console.rule("[bold blue]RAG Lab — Módulo 12.3: Rate Limiting")

    # Demo con middleware manual (funciona sin instalar nada extra)
    LIMIT = 5  # bajo para que la demo sea rápida
    app = build_app_manual(limit=LIMIT, window=60.0)
    client = TestClient(app)

    console.print(f"\n[bold]Simulando {LIMIT + 3} requests (límite: {LIMIT}/min)[/bold]\n")

    table = Table(title="Rate Limiting Demo — Ventana Deslizante", show_lines=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Status", width=8)
    table.add_column("Respuesta", style="white")

    for i in range(1, LIMIT + 4):
        resp = client.post("/query", json={"question": f"pregunta {i}"})
        status_color = "green" if resp.status_code == 200 else "red"
        table.add_row(
            str(i),
            f"[{status_color}]{resp.status_code}[/{status_color}]",
            str(resp.json()),
        )

    console.print(table)

    console.print(Panel(
        "[bold]slowapi[/bold] — pip install slowapi\n"
        "  • Decorador @limiter.limit('10/minute') por endpoint\n"
        "  • Clave por IP, API key, user ID o función custom\n"
        "  • Soporte para Redis como backend de contadores\n\n"
        "[bold]Middleware manual[/bold] — sin dependencias\n"
        "  • Ventana deslizante en dict de memoria\n"
        "  • No persiste entre reinicios ni entre workers\n"
        "  • Para prod: reemplazar dict por Redis con EXPIRE",
        title="Opciones de Rate Limiting",
        border_style="yellow",
    ))

    # Mostrar cómo sería con slowapi si está disponible
    try:
        from slowapi import Limiter  # noqa: F401
        console.print(Panel(
            "@app.post('/query')\n"
            "@limiter.limit('10/minute')\n"
            "def query(req: QueryRequest, request: Request):\n"
            "    ...\n\n"
            "Límites: '10/minute', '100/hour', '1000/day', '5/second'",
            title="slowapi instalado — uso con decorador",
            border_style="green",
        ))
    except ImportError:
        console.print("[dim]slowapi no instalado. pip install slowapi para el enfoque con decoradores.[/dim]")


if __name__ == "__main__":
    main()
