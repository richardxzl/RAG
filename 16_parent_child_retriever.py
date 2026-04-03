"""
16_parent_child_retriever.py — Módulo 3.3: Parent-Child Retriever

Estrategia: chunks pequeños para buscar con precisión,
chunks padres grandes para darle contexto completo al LLM.

Problema que resuelve: un chunk de 200 chars puede matchear bien la query
pero ser demasiado corto para que el LLM genere una buena respuesta.
La solución es buscar con el hijo pequeño pero responder con el padre grande.
"""
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.retriever import get_vectorstore
from rag.embeddings import get_embeddings
from rag.chain import get_llm, format_docs

console = Console()

QUERY = "¿De qué trata el documento?"


def build_parent_child_retriever():
    """
    child_splitter: chunks pequeños que se vectorizan y buscan (200 chars)
    parent_splitter: chunks grandes que se retornan al LLM (1000 chars)

    El vector store guarda los hijos. El docstore guarda los padres.
    Cuando un hijo hace match, el retriever busca su padre en el docstore.
    """
    # Vector store para los hijos (embeddings de chunks pequeños)
    vectorstore = get_vectorstore()

    # Docstore en memoria para los padres
    # En producción usarías Redis, Postgres, o cualquier key-value store
    docstore = InMemoryStore()

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever


def cargar_documentos_en_retriever(retriever: ParentDocumentRetriever):
    """
    Carga los documentos del vector store existente en el ParentDocumentRetriever.
    En producción esto se haría en el pipeline de ingesta (01_ingest.py).
    """
    from langchain_chroma import Chroma
    from rag.config import COLLECTION_NAME, CHROMA_DIR
    from langchain_core.documents import Document

    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"])
    docs_originales = [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(data["documents"], data["metadatas"])
    ]

    if docs_originales:
        retriever.add_documents(docs_originales[:10])  # subset para demo
    return docs_originales


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 3.3: Parent-Child Retriever")

    console.print("\n[dim]Construyendo retriever y cargando documentos...[/]")
    retriever = build_parent_child_retriever()
    docs_originales = cargar_documentos_en_retriever(retriever)

    if not docs_originales:
        console.print("[red]No hay documentos en ChromaDB. Ejecuta 01_ingest.py primero.[/]")
        return

    # Comparar: retriever normal vs parent-child
    from rag.retriever import get_retriever
    retriever_normal = get_retriever()

    docs_normal = retriever_normal.invoke(QUERY)
    docs_parent = retriever.invoke(QUERY)

    # Tabla comparativa de tamaño de chunks
    table = Table(title="Comparativa de chunks recuperados", show_header=True, header_style="bold magenta")
    table.add_column("Estrategia")
    table.add_column("Chunks")
    table.add_column("Longitud promedio")
    table.add_column("Longitud mínima")
    table.add_column("Longitud máxima")

    def stats(docs):
        lens = [len(d.page_content) for d in docs]
        if not lens:
            return 0, 0, 0
        return sum(lens) // len(lens), min(lens), max(lens)

    avg_n, min_n, max_n = stats(docs_normal)
    avg_p, min_p, max_p = stats(docs_parent)

    table.add_row("Normal (chunk_size=1000)", str(len(docs_normal)), f"{avg_n}", f"{min_n}", f"{max_n}")
    table.add_row("Parent-Child (hijo=200, padre=1000)", str(len(docs_parent)), f"{avg_p}", f"{min_p}", f"{max_p}")
    console.print(table)

    # Mostrar un chunk normal vs su equivalente parent
    if docs_normal:
        console.print(Panel(
            docs_normal[0].page_content[:300],
            title="Chunk normal (lo que busca Y lo que recibe el LLM)",
            border_style="dim",
        ))

    if docs_parent:
        console.print(Panel(
            docs_parent[0].page_content[:500],
            title="Chunk padre (mayor contexto para el LLM)",
            border_style="green",
        ))

    # Nota conceptual
    console.print(Panel(
        "[bold]Cómo funciona internamente:[/]\n\n"
        "1. Al indexar: el documento se divide en padres (1000 chars) y en hijos (200 chars).\n"
        "   Los hijos se vectorizan y guardan en ChromaDB.\n"
        "   Los padres se guardan en el docstore con un ID.\n\n"
        "2. Al buscar: el vector store encuentra los hijos más similares.\n"
        "   Para cada hijo, busca su padre en el docstore.\n"
        "   Retorna los padres (no los hijos) al LLM.\n\n"
        "[dim]El LLM recibe contexto amplio. La búsqueda fue precisa.[/]",
        title="Flujo interno",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
