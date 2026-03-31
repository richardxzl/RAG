"""
63_pgvector_migrate.py — Módulo 13.1: Migrar de ChromaDB a pgvector

Migra el vector store de ChromaDB (local) a Supabase pgvector (SQL).
Sin DATABASE_URL: modo demo con comparación y SQL de setup.

Requiere (para migración real):
  pip install langchain-postgres psycopg2-binary
  DATABASE_URL=postgresql://postgres.[ref]:[pass]@aws-0-[region].pooler.supabase.com:6543/postgres
"""
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

try:
    from langchain_postgres import PGVector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

from rag.retriever import get_vectorstore
from rag.embeddings import get_embeddings
from rag.config import COLLECTION_NAME


def get_pgvector_store(connection_string: str = None):
    """
    Crea o conecta al PGVector store en Supabase.
    La tabla se crea automáticamente si no existe.
    """
    if not PGVECTOR_AVAILABLE:
        raise ImportError("pip install langchain-postgres psycopg2-binary")

    conn = connection_string or os.getenv("DATABASE_URL")
    if not conn:
        raise ValueError("DATABASE_URL no configurada.")

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=conn,
    )


def migrate_chroma_to_pgvector(connection_string: str = None):
    """
    Lee todos los documentos de ChromaDB y los inserta en pgvector.
    Si hay embeddings pre-calculados los reutiliza (evita recalcular).
    Es idempotente si se usa collection_name única.
    """
    console.print("\n[bold]Paso 1: Leer documentos de ChromaDB[/bold]")
    chroma_vs = get_vectorstore()
    result = chroma_vs._collection.get(include=["documents", "metadatas", "embeddings"])

    total = len(result["documents"])
    console.print(f"  → {total} documentos encontrados")

    if total == 0:
        console.print("[yellow]  No hay documentos. Ejecuta 01_ingest.py primero.[/yellow]")
        return None

    console.print("\n[bold]Paso 2: Conectar a pgvector[/bold]")
    pgvector_vs = get_pgvector_store(connection_string)
    console.print("  → Conexión establecida")

    console.print("\n[bold]Paso 3: Insertar documentos[/bold]")
    from langchain_core.documents import Document

    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(result["documents"], result["metadatas"])
    ]

    if result.get("embeddings"):
        # Reutilizar embeddings pre-calculados — no recalcula, mucho más rápido
        pgvector_vs.add_embeddings(
            texts=[d.page_content for d in docs],
            embeddings=result["embeddings"],
            metadatas=[d.metadata for d in docs],
        )
    else:
        pgvector_vs.add_documents(docs)

    console.print(f"  → {total} documentos insertados en pgvector")
    return pgvector_vs


def show_demo():
    """Demo sin Supabase — comparación y SQL de setup."""
    console.rule("[bold blue]RAG Lab — Módulo 13.1: Migrar ChromaDB → pgvector")

    console.print(Panel(
        "[yellow]DATABASE_URL no configurada — modo demo[/yellow]\n\n"
        "Para la migración real:\n"
        "  1. Crear proyecto en supabase.com\n"
        "  2. SQL Editor → CREATE EXTENSION IF NOT EXISTS vector;\n"
        "  3. Copiar la connection string (Project Settings → Database)\n"
        "  4. pip install langchain-postgres psycopg2-binary\n"
        "  5. DATABASE_URL=... python 63_pgvector_migrate.py",
        title="Setup requerido",
        border_style="yellow",
    ))

    table = Table(title="ChromaDB vs pgvector", show_lines=True)
    table.add_column("Característica", style="cyan", width=22)
    table.add_column("ChromaDB", style="green", width=24)
    table.add_column("pgvector (Supabase)", style="blue", width=28)

    rows = [
        ("Storage", "Archivos locales", "PostgreSQL en la nube"),
        ("Escalabilidad", "Un proceso", "Horizontal con pgBouncer"),
        ("Backup", "Manual (copiar directorio)", "Automático"),
        ("Filtros metadata", "API propia", "SQL nativo (WHERE)"),
        ("Auth / RLS", "No", "Row Level Security nativo"),
        ("Full-text search", "No nativo", "ts_vector + pg_trgm"),
        ("Multitenancy", "Difícil", "Nativo con RLS"),
        ("Coste", "Gratis (local)", "Free tier disponible"),
        ("Índice", "HNSW por defecto", "HNSW o IVFFlat (configurable)"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)

    console.print(Panel(
        "-- 1. Activar la extensión (una vez por proyecto)\n"
        "CREATE EXTENSION IF NOT EXISTS vector;\n\n"
        "-- 2. langchain-postgres crea la tabla automáticamente\n"
        "-- Equivalente manual:\n"
        "CREATE TABLE IF NOT EXISTS langchain_pg_embedding (\n"
        "    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n"
        "    collection_id UUID,\n"
        "    embedding     vector(384),  -- igual que EMBEDDING_DIMS\n"
        "    document      TEXT,\n"
        "    cmetadata     JSONB\n"
        ");",
        title="SQL de setup en Supabase",
        border_style="blue",
    ))


def main():
    if not PGVECTOR_AVAILABLE or not os.getenv("DATABASE_URL"):
        show_demo()
        return

    console.rule("[bold blue]RAG Lab — Módulo 13.1: Migrar ChromaDB → pgvector")

    try:
        pgvector_vs = migrate_chroma_to_pgvector()
        if pgvector_vs:
            console.print("\n[bold]Verificación: búsqueda de prueba[/bold]")
            docs = pgvector_vs.similarity_search("LangChain", k=2)
            for i, doc in enumerate(docs, 1):
                console.print(f"  [{i}] {doc.page_content[:100]}...")

            console.print(Panel(
                "[green]Migración completada exitosamente.[/green]\n"
                "Cambia get_vectorstore() en rag/retriever.py para usar PGVector.",
                border_style="green",
            ))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
