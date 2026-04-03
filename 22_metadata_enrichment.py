"""
22_metadata_enrichment.py — Módulo 4.3: Metadata Enrichment

Demuestra cómo agregar metadata útil a cada chunk durante la ingesta.
La metadata rica habilita: filtrado preciso, Self-Query, fuentes más informativas.

Metadata que agregamos:
  - source, page (ya existe)
  - file_type, file_name (derivado del path)
  - chunk_index, total_chunks (posición dentro del documento)
  - char_count, word_count (tamaño del chunk)
  - section (detectada por heurística simple)
  - ingested_at (timestamp de ingesta)
"""
import os
import re
import logging
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.retriever import get_vectorstore

console = Console()


# ── Funciones de enriquecimiento ──────────────────────────────────────────────

def enrich_metadata(doc: Document, chunk_index: int, total_chunks: int) -> Document:
    """
    Agrega metadata calculada a un chunk. No modifica el contenido.
    Retorna un nuevo Document con la metadata extendida.
    """
    source = doc.metadata.get("source", "")
    existing_meta = doc.metadata.copy()

    enriched = {
        **existing_meta,
        # Información del archivo
        "file_name": os.path.basename(source) if source else "unknown",
        "file_type": os.path.splitext(source)[1].lstrip(".").lower() if source else "unknown",
        # Posición dentro del documento
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        # Estadísticas del chunk
        "char_count": len(doc.page_content),
        "word_count": len(doc.page_content.split()),
        # Sección detectada por heurística (primera línea si parece título)
        "section": detect_section(doc.page_content),
        # Timestamp de ingesta (ISO 8601, siempre UTC)
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    return Document(page_content=doc.page_content, metadata=enriched)


def detect_section(text: str) -> str:
    """
    Heurística simple: si la primera línea es corta y en mayúsculas o termina en ':',
    probablemente es un título de sección.
    """
    primera_linea = text.strip().split("\n")[0].strip()
    if len(primera_linea) < 80 and (
        primera_linea.endswith(":") or
        primera_linea.isupper() or
        re.match(r"^#+\s", primera_linea)   # Header markdown
    ):
        return primera_linea.lstrip("#").strip()
    return ""


def enrich_document_chunks(docs: list[Document]) -> list[Document]:
    """
    Dado una lista de chunks (del splitter), enriquece cada uno con metadata calculada.
    Agrupa por 'source' para calcular total_chunks por documento.
    """
    # Agrupar por fuente para contar chunks por documento
    por_fuente: dict[str, list] = {}
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "unknown")
        if src not in por_fuente:
            por_fuente[src] = []
        por_fuente[src].append((i, doc))

    enriquecidos = [None] * len(docs)
    for src, items in por_fuente.items():
        total = len(items)
        for chunk_idx, (original_idx, doc) in enumerate(items):
            enriquecidos[original_idx] = enrich_metadata(doc, chunk_idx, total)

    return enriquecidos


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_metadata_basica():
    """Muestra la metadata que tienen los chunks actuales en ChromaDB."""
    console.rule("[bold yellow]Metadata actual en ChromaDB (sin enriquecimiento)")

    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"], limit=3)

    for i, (content, meta) in enumerate(zip(data["documents"], data["metadatas"]), 1):
        console.print(Panel(
            JSON(json.dumps(meta or {}, ensure_ascii=False, indent=2)),
            title=f"Chunk {i} — metadata original",
            border_style="dim",
        ))

    console.print("[dim]Solo 'source' y 'page' — metadata mínima.[/]")


def demo_metadata_enriquecida():
    """Simula el proceso de enriquecimiento sobre docs del vector store."""
    console.rule("[bold yellow]Metadata enriquecida")

    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"], limit=5)
    docs = [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(data["documents"], data["metadatas"])
    ]

    if not docs:
        console.print("[yellow]No hay documentos en ChromaDB. Ejecuta 01_ingest.py primero.[/]")
        return

    enriquecidos = enrich_document_chunks(docs)

    # Mostrar el primero con metadata completa
    if enriquecidos:
        doc = enriquecidos[0]
        console.print(Panel(
            JSON(json.dumps(doc.metadata, ensure_ascii=False, indent=2)),
            title="Chunk enriquecido — metadata completa",
            border_style="green",
        ))

    # Tabla comparativa de todos
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("file_name", max_width=20)
    table.add_column("chunk_index")
    table.add_column("word_count")
    table.add_column("section", max_width=30)

    for i, doc in enumerate(enriquecidos[:5], 1):
        m = doc.metadata
        table.add_row(
            str(i),
            m.get("file_name", "?"),
            f"{m.get('chunk_index', '?')} / {m.get('total_chunks', '?')}",
            str(m.get("word_count", "?")),
            m.get("section", "—")[:28],
        )
    console.print(table)


def demo_valor_en_retrieval():
    """Muestra cómo la metadata enriquecida mejora las fuentes mostradas al usuario."""
    console.rule("[bold yellow]Impacto en la experiencia del usuario")

    console.print(Panel(
        "[bold]Sin metadata enriquecida:[/]\n"
        "  Fuentes utilizadas:\n"
        "  1. ./docs/manual.pdf (pág. 3)\n"
        "  2. ./docs/manual.pdf (pág. 3)\n"
        "  3. ./docs/manual.pdf (pág. 7)\n\n"
        "[bold]Con metadata enriquecida:[/]\n"
        "  Fuentes utilizadas:\n"
        "  1. manual.pdf — Sección: 'Instalación' (chunk 2/15, 145 palabras)\n"
        "  2. guia_inicio.md — Sección: 'LCEL — Ventajas' (chunk 8/23, 98 palabras)\n"
        "  3. manual.pdf — Sección: 'Configuración' (chunk 5/15, 201 palabras)\n\n"
        "[dim]La metadata no solo ayuda al retrieval — mejora la transparencia para el usuario.[/]",
        border_style="blue",
    ))


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 4.3: Metadata Enrichment")
    demo_metadata_basica()
    demo_metadata_enriquecida()
    demo_valor_en_retrieval()


if __name__ == "__main__":
    main()
