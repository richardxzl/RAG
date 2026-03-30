"""
23_chunk_size_comparison.py — Módulo 4.4: Comparar chunk sizes

Test automatizado que evalúa chunk_size 500 vs 1000 vs 1500.
Métricas por configuración:
  - Número de chunks generados
  - Distribución de tamaños (min, max, promedio)
  - Cobertura: ¿cuántos chunks distintos recupera cada query?
  - Solapamiento entre chunks recuperados (proxy de redundancia)
  - Tokens de contexto enviados al LLM (estimado)
"""
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import get_embeddings
from rag.retriever import get_vectorstore
from rag.config import COLLECTION_NAME, RETRIEVAL_K

console = Console()

CHUNK_SIZES = [500, 1000, 1500]
CHUNK_OVERLAP_RATIO = 0.2  # 20% de solapamiento siempre

QUERIES_EVAL = [
    "¿De qué trata el documento?",
    "¿Qué tecnologías se mencionan?",
    "Resume los puntos más importantes",
    "¿Qué es LangChain?",
]


def cargar_docs_originales() -> list[Document]:
    """Carga todos los documentos del vector store existente."""
    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(data["documents"], data["metadatas"])
    ]


def crear_vectorstore_temporal(docs: list[Document], chunk_size: int) -> Chroma:
    """
    Crea un vector store en memoria (sin persistencia) con el chunk_size dado.
    Usa collection_name único para no colisionar con el store principal.
    """
    overlap = int(chunk_size * CHUNK_OVERLAP_RATIO)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_documents(docs)

    # Vector store en memoria (no persiste)
    vs = Chroma(
        collection_name=f"eval_chunk_{chunk_size}",
        embedding_function=get_embeddings(),
    )
    if chunks:
        vs.add_documents(chunks)
    return vs, chunks


def evaluar_configuracion(docs_originales: list[Document], chunk_size: int) -> dict:
    """Evalúa una configuración de chunk_size y retorna métricas."""
    vs, chunks = crear_vectorstore_temporal(docs_originales, chunk_size)

    if not chunks:
        return {
            "chunk_size": chunk_size,
            "num_chunks": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_docs_per_query": 0,
            "avg_solapamiento": 0,
            "avg_tokens_contexto": 0,
            "latencia_ms": 0,
        }

    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    chars = [len(c.page_content) for c in chunks]
    docs_por_query = []
    solapamientos = []
    tokens_contexto = []
    tiempos = []

    for query in QUERIES_EVAL:
        t0 = time.time()
        docs_recuperados = retriever.invoke(query)
        elapsed = (time.time() - t0) * 1000

        tiempos.append(elapsed)
        docs_por_query.append(len(docs_recuperados))

        # Solapamiento entre chunks recuperados
        if len(docs_recuperados) > 1:
            palabras = [set(d.page_content.lower().split()) for d in docs_recuperados]
            overlaps = []
            for i in range(len(palabras)):
                for j in range(i + 1, len(palabras)):
                    inter = palabras[i] & palabras[j]
                    union = palabras[i] | palabras[j]
                    overlaps.append(len(inter) / len(union) if union else 0)
            solapamientos.append(sum(overlaps) / len(overlaps) * 100)

        # Estimación de tokens (aprox 4 chars = 1 token)
        total_chars = sum(len(d.page_content) for d in docs_recuperados)
        tokens_contexto.append(total_chars // 4)

    return {
        "chunk_size": chunk_size,
        "num_chunks": len(chunks),
        "avg_chars": sum(chars) // len(chars),
        "min_chars": min(chars),
        "max_chars": max(chars),
        "avg_docs_per_query": sum(docs_por_query) / len(docs_por_query),
        "avg_solapamiento": sum(solapamientos) / len(solapamientos) if solapamientos else 0,
        "avg_tokens_contexto": sum(tokens_contexto) // len(tokens_contexto),
        "latencia_ms": sum(tiempos) / len(tiempos),
    }


def main():
    console.rule("[bold blue]RAG Lab — Módulo 4.4: Comparar chunk sizes")

    console.print("\n[dim]Cargando documentos originales...[/]")
    docs_originales = cargar_docs_originales()

    if not docs_originales:
        console.print("[red]No hay documentos en ChromaDB. Ejecuta 01_ingest.py primero.[/]")
        return

    console.print(f"  {len(docs_originales)} chunks en el store → re-chunkeando con 3 configuraciones...\n")

    resultados = []
    for chunk_size in CHUNK_SIZES:
        console.print(f"[dim]Evaluando chunk_size={chunk_size}...[/]")
        r = evaluar_configuracion(docs_originales, chunk_size)
        resultados.append(r)

    # Tabla de distribución de chunks
    t1 = Table(title="Distribución de chunks por configuración", show_header=True, header_style="bold magenta")
    t1.add_column("chunk_size")
    t1.add_column("Total chunks")
    t1.add_column("Chars promedio")
    t1.add_column("Chars mín")
    t1.add_column("Chars máx")

    for r in resultados:
        t1.add_row(
            str(r["chunk_size"]),
            str(r["num_chunks"]),
            str(r["avg_chars"]),
            str(r["min_chars"]),
            str(r["max_chars"]),
        )
    console.print(t1)

    # Tabla de métricas de retrieval
    t2 = Table(title="Métricas de retrieval", show_header=True, header_style="bold cyan")
    t2.add_column("chunk_size")
    t2.add_column("Docs/query (k={})".format(RETRIEVAL_K))
    t2.add_column("Solapamiento ↓")
    t2.add_column("Tokens contexto ↓")
    t2.add_column("Latencia (ms)")

    for r in resultados:
        t2.add_row(
            str(r["chunk_size"]),
            f"{r['avg_docs_per_query']:.1f}",
            f"{r['avg_solapamiento']:.1f}%",
            str(r["avg_tokens_contexto"]),
            f"{r['latencia_ms']:.1f}",
        )
    console.print(t2)

    # Recomendación basada en los datos
    mejor_solapamiento = min(resultados, key=lambda r: r["avg_solapamiento"])
    menos_tokens = min(resultados, key=lambda r: r["avg_tokens_contexto"])

    console.print(Panel(
        f"[bold]Observaciones:[/]\n\n"
        f"· Menor solapamiento: chunk_size={mejor_solapamiento['chunk_size']} "
        f"({mejor_solapamiento['avg_solapamiento']:.1f}%) → más diversidad en los resultados\n"
        f"· Menos tokens de contexto: chunk_size={menos_tokens['chunk_size']} "
        f"({menos_tokens['avg_tokens_contexto']} tokens) → más barato por query\n\n"
        f"[dim]El chunk_size óptimo depende del dominio y del LLM usado.\n"
        f"Para respuestas precisas → chunk pequeño (500).\n"
        f"Para respuestas con contexto amplio → chunk grande (1500).\n"
        f"El módulo 5 (Evaluación) permite medir esto con calidad real.[/]",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
