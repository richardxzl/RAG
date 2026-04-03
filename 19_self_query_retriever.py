"""
19_self_query_retriever.py — Módulo 3.6: Self-Query Retriever

El LLM convierte el query en lenguaje natural + filtros de metadata.
Ejemplo: "documentos de 2024 sobre Python" →
  query semántico: "Python"
  filtro metadata: {"año": {"$eq": 2024}}

Requiere que los documentos tengan metadata estructurada.
"""
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON
import json

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.documents import Document

from rag.retriever import get_vectorstore
from rag.chain import get_llm

console = Console()


# ── Metadata schema — qué campos tiene cada documento ────────────────────────

# Definimos los atributos de metadata que el LLM puede usar como filtros
METADATA_FIELD_INFO = [
    AttributeInfo(
        name="source",
        description="Nombre del archivo fuente del documento",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="Número de página dentro del documento",
        type="integer",
    ),
]

DOCUMENT_CONTENT_DESCRIPTION = (
    "Fragmentos de documentos técnicos sobre LangChain, RAG y arquitecturas de IA"
)


def build_self_query_retriever():
    """
    SelfQueryRetriever convierte el query natural en:
      - query semántico (para el vector store)
      - filtros de metadata (para ChromaDB)
    """
    vectorstore = get_vectorstore()
    llm = get_llm()

    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=METADATA_FIELD_INFO,
        verbose=True,   # loguea el query estructurado generado
    )


def mostrar_resultado(query: str, docs: list):
    console.print(f"\n[bold]Query:[/] {query}")
    console.print(f"  Docs recuperados: {len(docs)}")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        console.print(Panel(
            f"[dim]Metadata:[/] {json.dumps(meta, ensure_ascii=False)}\n\n"
            + doc.page_content[:200].replace("\n", " "),
            title=f"Doc {i}",
            border_style="green",
        ))


def demo_sin_self_query():
    """Muestra el comportamiento sin filtros de metadata para comparar."""
    console.rule("[bold yellow]Sin Self-Query (solo similitud semántica)")
    from rag.retriever import get_retriever
    retriever = get_retriever()
    docs = retriever.invoke("documentos sobre LangChain")
    mostrar_resultado("documentos sobre LangChain", docs)


def demo_con_self_query():
    """Con Self-Query, el LLM puede filtrar por metadata además de buscar semánticamente."""
    console.rule("[bold yellow]Con Self-Query Retriever")

    retriever = build_self_query_retriever()

    queries = [
        "¿De qué trata el documento?",
        "Contenido de la primera página",
    ]

    for q in queries:
        try:
            docs = retriever.invoke(q)
            mostrar_resultado(q, docs)
        except Exception as e:
            console.print(f"  [yellow]Query '{q}': {e}[/]")
            console.print("  [dim](El LLM no detectó filtros de metadata — consulta semántica pura)[/]")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 3.6: Self-Query Retriever")

    console.print(Panel(
        "[bold]Cómo funciona:[/]\n\n"
        "El LLM analiza el query en lenguaje natural y extrae:\n"
        "  1. La parte semántica → búsqueda vectorial\n"
        "  2. Los filtros de metadata → filtrado en ChromaDB\n\n"
        "Ejemplo:\n"
        "  Query: 'documentos sobre Python de la página 3'\n"
        "  → Semántico: 'Python'\n"
        "  → Filtro: {\"page\": {\"$eq\": 3}}\n\n"
        "[dim]Esto permite queries más precisos sin necesidad de "
        "que el usuario conozca la estructura interna de la DB.[/]",
        title="Concepto",
        border_style="blue",
    ))

    # Mostrar la metadata disponible en los documentos
    console.print("\n[bold]Metadata disponible en los documentos:[/]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Campo")
    table.add_column("Tipo")
    table.add_column("Descripción")
    for field in METADATA_FIELD_INFO:
        table.add_row(field.name, field.type, field.description)
    console.print(table)

    demo_sin_self_query()
    demo_con_self_query()

    console.print(Panel(
        "[bold]Limitación importante:[/]\n\n"
        "Self-Query solo es útil si los documentos tienen metadata RICA y ESTRUCTURADA.\n"
        "Si los documentos solo tienen 'source' y 'page', el valor es limitado.\n\n"
        "Para sacarle partido real, el pipeline de ingesta (01_ingest.py) debería\n"
        "agregar metadata como: autor, fecha, tema, categoría, idioma, etc.\n"
        "Eso se verá en el Módulo 4.3 (Metadata Enrichment).",
        border_style="yellow",
    ))


if __name__ == "__main__":
    main()
