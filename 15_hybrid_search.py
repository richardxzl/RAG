"""
15_hybrid_search.py — Módulo 3.2: Hybrid Search (BM25 + embeddings)

Combina dos tipos de retrieval complementarios:
  - BM25: búsqueda por palabras exactas (TF-IDF moderno), ideal para términos técnicos
  - Embeddings (ChromaDB): búsqueda semántica, ideal para paráfrasis y sinónimos

EnsembleRetriever fusiona ambos resultados con Reciprocal Rank Fusion (RRF).
"""
import logging
from rich.console import Console
from rich.table import Table

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from rag.retriever import get_vectorstore, get_retriever
from rag.config import RETRIEVAL_K

console = Console()

QUERIES = [
    "¿De qué trata el documento?",           # semántica → embeddings gana
    "LangChain LCEL pipeline",               # términos exactos → BM25 gana
]


def cargar_docs_para_bm25() -> list[Document]:
    """
    BM25Retriever necesita los documentos en memoria (no usa un vector store).
    Los cargamos directamente del vector store de ChromaDB.
    """
    vs = get_vectorstore()
    # get() retorna todos los documentos del collection
    data = vs.get(include=["documents", "metadatas"])
    docs = []
    for content, meta in zip(data["documents"], data["metadatas"]):
        docs.append(Document(page_content=content, metadata=meta or {}))
    return docs


def build_hybrid_retriever(k: int = RETRIEVAL_K, bm25_weight: float = 0.4):
    """
    bm25_weight: peso del BM25 (0-1). El resto va a embeddings.
    0.4/0.6 es un buen punto de partida — embeddings suelen ser más fuertes.
    """
    docs = cargar_docs_para_bm25()

    bm25 = BM25Retriever.from_documents(docs, k=k)
    embedding = get_retriever()

    return EnsembleRetriever(
        retrievers=[bm25, embedding],
        weights=[bm25_weight, 1 - bm25_weight],
    )


def mostrar_comparacion(query: str, docs_embedding: list, docs_hybrid: list):
    console.print(f"\n[bold]Query:[/] {query}")

    t = Table(show_header=True, header_style="bold magenta", expand=True)
    t.add_column("Solo Embeddings", max_width=50)
    t.add_column("Hybrid (BM25 + Embeddings)", max_width=50)

    max_rows = max(len(docs_embedding), len(docs_hybrid))
    for i in range(max_rows):
        col1 = docs_embedding[i].page_content[:80].replace("\n", " ") if i < len(docs_embedding) else ""
        col2 = docs_hybrid[i].page_content[:80].replace("\n", " ") if i < len(docs_hybrid) else ""
        t.add_row(col1, col2)

    console.print(t)

    # Diferencias
    textos_emb = {d.page_content[:60] for d in docs_embedding}
    textos_hyb = {d.page_content[:60] for d in docs_hybrid}
    solo_en_hybrid = textos_hyb - textos_emb
    if solo_en_hybrid:
        console.print(f"  [cyan]Docs exclusivos del hybrid:[/] {len(solo_en_hybrid)} "
                      f"(aportados por BM25)")
    else:
        console.print("  [dim]Mismos resultados en ambos (el corpus es pequeño)[/]")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 3.2: Hybrid Search")

    console.print("\n[dim]Cargando documentos para BM25...[/]")
    hybrid = build_hybrid_retriever()
    embedding_only = get_retriever()

    for query in QUERIES:
        docs_emb = embedding_only.invoke(query)
        docs_hyb = hybrid.invoke(query)
        mostrar_comparacion(query, docs_emb, docs_hyb)

    # Tabla conceptual de cuándo cada uno gana
    console.print()
    t2 = Table(title="Cuándo cada retriever gana", show_header=True, header_style="bold cyan")
    t2.add_column("Tipo de query", style="bold")
    t2.add_column("BM25")
    t2.add_column("Embeddings")
    t2.add_column("Hybrid")

    t2.add_row("Términos técnicos exactos (API, UUID)", "✅ Gana", "⚠️ Puede fallar", "✅")
    t2.add_row("Paráfrasis / sinónimos", "❌ Falla", "✅ Gana", "✅")
    t2.add_row("Preguntas cortas ambiguas", "⚠️", "✅", "✅")
    t2.add_row("Nombres propios / acrónimos", "✅ Gana", "⚠️", "✅")
    t2.add_row("Preguntas semánticas largas", "❌", "✅ Gana", "✅")
    console.print(t2)


if __name__ == "__main__":
    main()
