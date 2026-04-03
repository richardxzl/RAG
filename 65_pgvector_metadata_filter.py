"""
65_pgvector_metadata_filter.py — Módulo 13.3: Filtrado por metadata en SQL

pgvector + PostgreSQL permite filtrar documentos con SQL completo:
JOINs, full-text, window functions, RLS — imposible en ChromaDB.
No requiere conexión a Supabase — los ejemplos son educativos.
"""
import os
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

try:
    from langchain_postgres import PGVector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

from rag.embeddings import get_embeddings
from rag.config import COLLECTION_NAME


# ── Filtros con la API de LangChain ──────────────────────────

def demo_langchain_filters():
    """Muestra cómo usar filtros con langchain-postgres."""
    if not PGVECTOR_AVAILABLE or not os.getenv("DATABASE_URL"):
        return

    pgvector = PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=os.getenv("DATABASE_URL"),
    )

    # Igualdad simple
    docs = pgvector.similarity_search(
        "¿Qué es LangChain?",
        k=4,
        filter={"source": "langchain_docs.pdf"},
    )

    # Operadores de comparación
    docs = pgvector.similarity_search(
        "configuración",
        k=4,
        filter={"chunk_index": {"$gte": 5, "$lte": 20}},
    )

    # OR entre fuentes
    docs = pgvector.similarity_search(
        "memoria",
        k=4,
        filter={"$or": [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]},
    )

    return docs


# ── SQL de ejemplo ────────────────────────────────────────────

SQL_EXAMPLES = {
    "Filtro básico por source": """\
SELECT document, cmetadata,
       1 - (embedding <=> '[vector]'::vector) AS similarity
FROM langchain_pg_embedding
WHERE collection_id = $1
  AND cmetadata->>'source' = 'langchain_docs.pdf'
ORDER BY embedding <=> '[vector]'::vector
LIMIT 4;""",

    "Full-text + vectorial combinados": """\
-- Combina similitud vectorial (70%) con relevancia full-text (30%)
SELECT document, cmetadata,
       (1 - (embedding <=> '[vector]'::vector)) * 0.7
       + ts_rank(to_tsvector('spanish', document),
                 plainto_tsquery('spanish', 'LangChain memoria')) * 0.3 AS score
FROM langchain_pg_embedding
WHERE collection_id = $1
  AND to_tsvector('spanish', document)
      @@ plainto_tsquery('spanish', 'LangChain memoria')
ORDER BY score DESC
LIMIT 4;""",

    "Deduplicar por fuente (top-1 por doc)": """\
-- Evita que un documento monopolice los k resultados
WITH ranked AS (
    SELECT document, cmetadata,
           embedding <=> '[vector]'::vector AS dist,
           ROW_NUMBER() OVER (
               PARTITION BY cmetadata->>'source'
               ORDER BY embedding <=> '[vector]'::vector
           ) AS rn
    FROM langchain_pg_embedding
    WHERE collection_id = $1
)
SELECT document, cmetadata, dist
FROM ranked
WHERE rn = 1
ORDER BY dist
LIMIT 4;""",

    "Row Level Security (multi-tenant)": """\
-- Activar RLS: cada usuario solo ve sus documentos
ALTER TABLE langchain_pg_embedding ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_isolation ON langchain_pg_embedding
    USING (cmetadata->>'user_id' = current_user);

-- Con esto, el mismo RAG sirve a múltiples usuarios
-- sin cambiar nada en el código Python.""",
}

FILTER_OPERATORS = {
    "$eq": "Igual (default para valores simples)",
    "$ne": "No igual",
    "$gt / $gte": "Mayor que / mayor o igual",
    "$lt / $lte": "Menor que / menor o igual",
    "$in": "El valor está en la lista",
    "$nin": "El valor NO está en la lista",
    "$and": "AND lógico (default cuando hay múltiples claves)",
    "$or": "OR lógico",
}


def show_comparison():
    console.rule("[bold blue]RAG Lab — Módulo 13.3: Filtrado por Metadata en SQL")

    table = Table(title="ChromaDB vs pgvector — Capacidades de Filtrado", show_lines=True)
    table.add_column("Capacidad", style="cyan", width=28)
    table.add_column("ChromaDB", style="green", width=12)
    table.add_column("pgvector", style="blue", width=12)

    rows = [
        ("Filtro por igualdad", "✓", "✓"),
        ("Operadores ($gt, $lt...)", "✓", "✓"),
        ("AND / OR combinados", "✓", "✓"),
        ("Full-text search", "✗", "✓ (ts_vector)"),
        ("Rango de fechas", "Limitado", "✓ (CAST)"),
        ("Arrays en metadata", "✗", "✓ (? operator)"),
        ("JOINs con otras tablas", "✗", "✓"),
        ("Deduplicación por campo", "✗", "✓ (ROW_NUMBER)"),
        ("Regex en documentos", "✗", "✓ (~ operator)"),
        ("Row Level Security", "✗", "✓ nativo"),
        ("Subqueries", "✗", "✓"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)


def show_langchain_api():
    console.print(Panel(
        "from langchain_postgres import PGVector\n\n"
        "retriever = pgvector.as_retriever(\n"
        "    search_kwargs={\n"
        "        'k': 4,\n"
        "        'filter': {'source': 'doc.pdf'},  # igualdad\n"
        "    }\n"
        ")\n\n"
        "# Operadores de comparación\n"
        "filter = {\n"
        "    'chunk_index': {'$gte': 5, '$lte': 20},\n"
        "    'file_type': {'$in': ['pdf', 'md']},\n"
        "}\n\n"
        "# OR entre fuentes\n"
        "filter = {\n"
        "    '$or': [\n"
        "        {'source': 'doc1.pdf'},\n"
        "        {'source': 'doc2.pdf'},\n"
        "    ]\n"
        "}",
        title="API de filtros LangChain",
        border_style="green",
    ))


def show_operators():
    table = Table(title="Operadores de Filtro (LangChain → SQL)", show_lines=True)
    table.add_column("Operador", style="cyan", width=14)
    table.add_column("Descripción", style="white")
    for op, desc in FILTER_OPERATORS.items():
        table.add_row(op, desc)
    console.print(table)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    show_comparison()
    show_langchain_api()
    show_operators()

    console.print("\n[bold]SQL avanzado — acceso directo a PostgreSQL[/bold]")
    for title, sql in SQL_EXAMPLES.items():
        console.print(Panel(sql, title=title, border_style="blue"))

    console.print(Panel(
        "La ventaja real de pgvector no son los filtros básicos (esos ChromaDB también los tiene)\n"
        "sino el acceso al SQL completo de PostgreSQL:\n\n"
        "• Full-text search combinado con vectores en una sola query\n"
        "• Row Level Security para multi-tenant sin código extra\n"
        "• JOINs con tablas de tu app (users, projects, permissions)\n"
        "• Window functions para deduplicar y rankear\n"
        "• Auditoría nativa con triggers y pg_audit",
        title="Cuándo migrar a pgvector",
        border_style="yellow",
    ))


if __name__ == "__main__":
    main()
