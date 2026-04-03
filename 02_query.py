"""
02_query.py — Consulta RAG con LCEL + Cache

Entry point simple que usa los módulos de rag/.
"""
import sys
import logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from rag.chain import build_query_chain

console = Console()


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Consulta (LCEL + Cache)")

    query_fn, semantic_cache = build_query_chain()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "¿De qué trata el documento?"

    console.print(f"\n🔍 Pregunta: [bold]{question}[/]")
    console.print("   Buscando en la base de conocimiento...")

    answer, source_docs, cache_hit = query_fn(question)

    if cache_hit:
        console.print("   ⚡ [yellow]Respuesta desde cache semántico[/]")

    console.print(Panel(
        Markdown(answer),
        title="💬 Respuesta",
        border_style="green",
    ))

    console.print("\n📚 [bold]Fuentes utilizadas:[/]")
    for i, doc in enumerate(source_docs):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        console.print(f"   {i+1}. {source} (pág. {page})")
        console.print(f"      [dim]{doc.page_content[:100]}...[/]")

    # Stats del cache
    stats = semantic_cache.stats()
    console.print(f"\n📊 Cache: {stats['entries']} entradas, TTL={stats['ttl_seconds']}s, threshold={stats['threshold']}")

    if len(sys.argv) <= 1:
        console.print("\n💡 Tip: python 02_query.py '¿tu pregunta aquí?'")


if __name__ == "__main__":
    main()
