"""
59_fastapi_wrapper.py — Módulo 12.1: FastAPI wrapper del RAG

Envuelve el RAG en una API REST con FastAPI.
Demo: python 59_fastapi_wrapper.py
Producción: uvicorn 59_fastapi_wrapper:app --reload --port 8000
"""
import json
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.chain import build_query_chain

console = Console()

# ── Modelos Pydantic ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    content: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    cache_hit: bool

# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="RAG API",
    description="API REST sobre LangChain + ChromaDB",
    version="1.0.0",
)

# Inicializar la chain una sola vez al arrancar (evita cold start por request)
_query_fn, _cache = build_query_chain()


@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Consulta al RAG."""
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="La pregunta no puede estar vacía.")

    answer, docs, cache_hit = _query_fn(req.question)

    sources = [
        SourceDoc(
            content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            source=doc.metadata.get("source", "desconocido"),
        )
        for doc in docs
    ]

    return QueryResponse(answer=answer, sources=sources, cache_hit=cache_hit)


# ── Demo ──────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]RAG Lab — Módulo 12.1: FastAPI Wrapper")

    client = TestClient(app)

    # Liveness check
    console.print("\n[bold]GET /health[/bold]")
    resp = client.get("/health")
    console.print(Panel(json.dumps(resp.json(), indent=2), title=f"Status {resp.status_code}", border_style="green"))

    # Query exitosa
    console.print("\n[bold]POST /query[/bold]")
    resp = client.post("/query", json={"question": "¿Qué es LangChain?"})
    data = resp.json()

    table = Table(title="Respuesta del RAG", show_lines=True)
    table.add_column("Campo", style="cyan", width=12)
    table.add_column("Valor", style="white")
    answer_preview = data["answer"][:300] + "..." if len(data["answer"]) > 300 else data["answer"]
    table.add_row("answer", answer_preview)
    table.add_row("cache_hit", str(data["cache_hit"]))
    table.add_row("sources", f"{len(data['sources'])} documentos")
    console.print(table)

    # Error: pregunta vacía
    console.print("\n[bold]POST /query — pregunta vacía (espera 422)[/bold]")
    resp = client.post("/query", json={"question": ""})
    console.print(Panel(json.dumps(resp.json(), indent=2), title=f"Status {resp.status_code}", border_style="red"))

    console.print(Panel(
        "[bold]uvicorn 59_fastapi_wrapper:app --reload --port 8000[/bold]\n\n"
        "Docs interactivos: http://localhost:8000/docs\n"
        "ReDoc:             http://localhost:8000/redoc",
        title="Arrancar en producción",
        border_style="yellow",
    ))


if __name__ == "__main__":
    main()
