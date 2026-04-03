"""
60_streaming_endpoint.py — Módulo 12.2: Streaming endpoint (SSE)

Endpoint de streaming con Server-Sent Events.
El cliente recibe tokens del LLM en tiempo real, igual que ChatGPT.
Demo: python 60_streaming_endpoint.py
"""
import asyncio
import json
import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()

# ── App ───────────────────────────────────────────────────────

app = FastAPI(title="RAG Streaming API")

_retriever = get_retriever()
_llm = get_llm()


class QueryRequest(BaseModel):
    question: str


async def sse_generator(question: str):
    """
    Generador async que emite eventos SSE token a token.
    Formato: data: {json}\n\n
    """
    docs = _retriever.invoke(question)
    context = format_docs(docs)
    prompt = QUERY_PROMPT.format_messages(context=context, question=question)

    # Primero emitir las fuentes
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # Luego los tokens del LLM uno a uno
    async for chunk in _llm.astream(prompt):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming endpoint — emite tokens SSE."""
    return StreamingResponse(
        sse_generator(req.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # evita que nginx bufferice la respuesta
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Demo directo (sin servidor HTTP) ─────────────────────────

async def demo_streaming():
    console.rule("[bold blue]RAG Lab — Módulo 12.2: Streaming Endpoint (SSE)")

    question = "¿Qué es LangChain?"
    console.print(f"\n[bold]Streaming directo:[/bold] [yellow]{question}[/yellow]\n")

    docs = _retriever.invoke(question)
    context = format_docs(docs)
    prompt = QUERY_PROMPT.format_messages(context=context, question=question)

    full_response = ""
    with Live(console=console, refresh_per_second=20) as live:
        async for chunk in _llm.astream(prompt):
            if chunk.content:
                full_response += chunk.content
                live.update(Text(full_response, style="white"))

    sources = list({doc.metadata.get("source", "unknown") for doc in docs})
    console.print(f"\n[dim]Fuentes: {', '.join(sources)}[/dim]")

    console.print(Panel(
        "Formato SSE emitido por el endpoint:\n\n"
        'data: {"type": "sources", "sources": ["doc.pdf"]}\n\n'
        'data: {"type": "token", "content": "La "}\n\n'
        'data: {"type": "token", "content": "respuesta"}\n\n'
        'data: {"type": "done"}\n\n'
        "---\n"
        "Consumir con curl:\n"
        "  curl -X POST http://localhost:8000/query/stream \\\n"
        "    -H 'Content-Type: application/json' \\\n"
        "    -d '{\"question\": \"¿Qué es LangChain?\"}' --no-buffer\n\n"
        "Consumir con JavaScript:\n"
        "  const res = await fetch('/query/stream', {...});\n"
        "  for await (const chunk of res.body) { /* tokens */ }",
        title="Protocolo SSE",
        border_style="yellow",
    ))


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    asyncio.run(demo_streaming())


if __name__ == "__main__":
    main()
