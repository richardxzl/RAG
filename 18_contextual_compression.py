"""
18_contextual_compression.py — Módulo 3.5: Contextual Compression

Filtra o comprime los chunks DESPUÉS de recuperarlos, eliminando
las partes irrelevantes para el query específico.

Problema: un chunk de 1000 chars puede ser relevante, pero solo 200 chars
de él responden la pregunta. El LLM recibe 800 chars de ruido.

Estrategias demostradas:
  1. LLMChainExtractor: el LLM extrae solo la parte relevante de cada chunk
  2. EmbeddingsFilter: elimina chunks cuya similitud con el query es baja
"""
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.retriever import get_retriever
from rag.embeddings import get_embeddings
from rag.chain import get_llm, QUERY_PROMPT, format_docs

console = Console()

QUERY = "¿Qué es LCEL y para qué sirve?"


# ── Estrategia 1: LLMChainExtractor ──────────────────────────────────────────

def build_llm_extractor():
    """
    Para cada chunk recuperado, llama al LLM con:
    'Dado este contexto y este query, extrae solo la parte relevante'.
    Retorna el extracto, no el chunk completo.
    """
    llm = get_llm()
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=get_retriever(),
    )


# ── Estrategia 2: EmbeddingsFilter ────────────────────────────────────────────

def build_embeddings_filter(threshold: float = 0.76):
    """
    Elimina chunks cuya similitud coseno con el query sea menor que threshold.
    Más rápido que LLMChainExtractor (no llama al LLM), pero menos inteligente.
    threshold: 0.75-0.80 es un rango razonable para embeddings de 384 dims.
    """
    compressor = EmbeddingsFilter(
        embeddings=get_embeddings(),
        similarity_threshold=threshold,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=get_retriever(),
    )


def mostrar_docs_comparacion(titulo: str, docs: list, color: str = "dim"):
    console.print(f"\n[bold]{titulo}[/] — {len(docs)} chunks")
    for i, doc in enumerate(docs, 1):
        console.print(Panel(
            doc.page_content[:300].replace("\n", " "),
            title=f"Chunk {i} ({len(doc.page_content)} chars)",
            border_style=color,
        ))


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 3.5: Contextual Compression")
    console.print(f"\n[bold]Query:[/] {QUERY}\n")

    # Sin compresión (baseline)
    retriever_base = get_retriever()
    docs_base = retriever_base.invoke(QUERY)
    total_chars_base = sum(len(d.page_content) for d in docs_base)
    mostrar_docs_comparacion("Sin compresión (baseline)", docs_base, "dim")

    # EmbeddingsFilter (rápido, sin LLM extra)
    console.print("\n[dim]Aplicando EmbeddingsFilter...[/]")
    retriever_emb = build_embeddings_filter(threshold=0.76)
    docs_emb = retriever_emb.invoke(QUERY)
    total_chars_emb = sum(len(d.page_content) for d in docs_emb)
    mostrar_docs_comparacion("EmbeddingsFilter (threshold=0.76)", docs_emb, "yellow")

    # LLMChainExtractor (inteligente, llama al LLM por chunk)
    console.print("\n[dim]Aplicando LLMChainExtractor (llama al LLM por cada chunk)...[/]")
    retriever_llm = build_llm_extractor()
    docs_llm = retriever_llm.invoke(QUERY)
    total_chars_llm = sum(len(d.page_content) for d in docs_llm)
    mostrar_docs_comparacion("LLMChainExtractor", docs_llm, "green")

    # Tabla resumen
    table = Table(title="Comparativa de compresión", show_header=True, header_style="bold magenta")
    table.add_column("Estrategia")
    table.add_column("Chunks devueltos")
    table.add_column("Total chars")
    table.add_column("Reducción")
    table.add_column("Llamadas LLM extra")

    def reduccion(total):
        if total_chars_base == 0:
            return "0%"
        return f"{(1 - total / total_chars_base) * 100:.0f}%"

    table.add_row("Sin compresión", str(len(docs_base)), str(total_chars_base), "0%", "0")
    table.add_row("EmbeddingsFilter", str(len(docs_emb)), str(total_chars_emb), reduccion(total_chars_emb), "0")
    table.add_row("LLMChainExtractor", str(len(docs_llm)), str(total_chars_llm), reduccion(total_chars_llm), str(len(docs_base)))
    console.print(table)


if __name__ == "__main__":
    main()
