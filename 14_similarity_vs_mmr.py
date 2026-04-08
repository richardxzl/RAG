"""
14_similarity_vs_mmr.py — Módulo 3.1: Similarity vs MMR

Compara dos estrategias de búsqueda en el vector store:
  - similarity: retorna los K chunks más similares (pueden ser redundantes)
  - mmr (Maximum Marginal Relevance): balancea similitud + diversidad

MMR evita que el retriever devuelva 4 chunks que dicen casi lo mismo.
"""
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from rag.retriever import get_vectorstore
from rag.config import RETRIEVAL_K

console = Console()

QUERIES = [
    "¿Qué es Python y para qué se usa?",
    "¿Cómo funciona Vue.js y sus componentes?",
    "¿Qué es Laravel y cómo maneja las rutas?",
    "¿Cómo escalar una aplicación Node.js?",
    "arquitectura de inteligencia artificial y agentes",
]
QUERY = QUERIES[0]  # cambia el índice para probar distintas queries


def get_retriever_similarity(k: int = RETRIEVAL_K):
    vs = get_vectorstore()
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def get_retriever_mmr(k: int = RETRIEVAL_K, fetch_k: int = 20, lambda_mult: float = 0.5):
    """
    fetch_k: cuántos candidatos recuperar antes de aplicar MMR (debe ser > k)
    lambda_mult: 0.0 = máxima diversidad, 1.0 = máxima similitud (equivale a similarity)
    """
    vs = get_vectorstore()
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


def mostrar_docs(docs: list, titulo: str):
    table = Table(title=titulo, show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Fuente", width=20)
    table.add_column("Contenido (primeros 120 chars)", max_width=60)

    for i, doc in enumerate(docs, 1):
        fuente = doc.metadata.get("source", "?").split("/")[-1]
        table.add_row(str(i), fuente, doc.page_content[:120].replace("\n", " "))

    console.print(table)


def calcular_solapamiento(docs: list) -> float:
    """Porcentaje de tokens compartidos entre chunks (proxy de redundancia)."""
    if len(docs) < 2:
        return 0.0
    palabras = [set(d.page_content.lower().split()) for d in docs]
    solapamientos = []
    for i in range(len(palabras)):
        for j in range(i + 1, len(palabras)):
            interseccion = palabras[i] & palabras[j]
            union = palabras[i] | palabras[j]
            solapamientos.append(len(interseccion) / len(union) if union else 0)
    return sum(solapamientos) / len(solapamientos) * 100


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 3.1: Similarity vs MMR")
    console.print(f"\n[bold]Query:[/] {QUERY}\n")

    # Similarity
    retriever_sim = get_retriever_similarity()
    docs_sim = retriever_sim.invoke(QUERY)
    solapamiento_sim = calcular_solapamiento(docs_sim)
    mostrar_docs(docs_sim, f"Similarity — {len(docs_sim)} docs")
    console.print(f"  Solapamiento promedio entre chunks: [yellow]{solapamiento_sim:.1f}%[/]\n")

    # MMR balanceado
    retriever_mmr = get_retriever_mmr(lambda_mult=0.5)
    docs_mmr = retriever_mmr.invoke(QUERY)
    solapamiento_mmr = calcular_solapamiento(docs_mmr)
    mostrar_docs(docs_mmr, f"MMR (λ=0.5) — {len(docs_mmr)} docs")
    console.print(f"  Solapamiento promedio entre chunks: [yellow]{solapamiento_mmr:.1f}%[/]\n")

    # MMR con máxima diversidad
    retriever_div = get_retriever_mmr(lambda_mult=0.1)
    docs_div = retriever_div.invoke(QUERY)
    solapamiento_div = calcular_solapamiento(docs_div)
    mostrar_docs(docs_div, f"MMR (λ=0.1, máx diversidad) — {len(docs_div)} docs")
    console.print(f"  Solapamiento promedio entre chunks: [yellow]{solapamiento_div:.1f}%[/]\n")

    # Resumen
    resumen = Table(title="Resumen comparativo", show_header=True, header_style="bold cyan")
    resumen.add_column("Estrategia")
    resumen.add_column("Solapamiento ↓ mejor")
    resumen.add_column("Cuándo usar")

    resumen.add_row("similarity", f"{solapamiento_sim:.1f}%", "Preguntas específicas, chunk único ideal")
    resumen.add_row("mmr λ=0.5", f"{solapamiento_mmr:.1f}%", "Balance — recomendado por defecto")
    resumen.add_row("mmr λ=0.1", f"{solapamiento_div:.1f}%", "Resúmenes, necesitas cubrir múltiples ángulos")
    console.print(resumen)


if __name__ == "__main__":
    main()
