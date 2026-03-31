"""
64_pgvector_indices.py — Módulo 13.2: Índices y performance en pgvector

HNSW vs IVFFlat: cuándo usar cada uno, cómo configurarlos y SQL de creación.
No requiere conexión a Supabase — todo el contenido es educativo.
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ── SQL de creación de índices ────────────────────────────────

HNSW_SQL = """\
-- HNSW: Hierarchical Navigable Small World
-- Recomendado para producción general (<5M vectores)
CREATE INDEX IF NOT EXISTS idx_embedding_hnsw
ON langchain_pg_embedding
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,               -- conexiones por nodo (4-64, default 16)
    ef_construction = 64  -- precisión al construir (>= 2*m, default 64)
);

-- Ajustar precisión en tiempo de búsqueda (mayor = más preciso, más lento)
SET hnsw.ef_search = 100;"""

IVFFLAT_SQL = """\
-- IVFFlat: Inverted File with Flat quantization
-- Para datasets masivos (>1M vectores)
CREATE INDEX IF NOT EXISTS idx_embedding_ivfflat
ON langchain_pg_embedding
USING ivfflat (embedding vector_cosine_ops)
WITH (
    lists = 100  -- √N donde N = número total de vectores
                 -- Para 10K docs → 100, para 1M docs → 1000
);

-- Ajustar en búsqueda (más probes = más preciso, más lento)
SET ivfflat.probes = 10;"""

METADATA_GIN_SQL = """\
-- Índice GIN para filtrado rápido sobre cmetadata (JSONB)
CREATE INDEX IF NOT EXISTS idx_metadata_gin
ON langchain_pg_embedding
USING gin (cmetadata);

-- Índice específico para un campo de metadata muy consultado
CREATE INDEX IF NOT EXISTS idx_metadata_source
ON langchain_pg_embedding ((cmetadata->>'source'));"""

PARTIAL_SQL = """\
-- Índice parcial: solo para una colección específica
-- Útil si tienes múltiples colecciones en la misma tabla
CREATE INDEX IF NOT EXISTS idx_embedding_collection_hnsw
ON langchain_pg_embedding
USING hnsw (embedding vector_cosine_ops)
WHERE collection_id = (
    SELECT uuid FROM langchain_pg_collection
    WHERE name = 'mi_knowledge_base'
);"""


def show_comparison():
    console.rule("[bold blue]RAG Lab — Módulo 13.2: Índices y Performance")

    table = Table(title="HNSW vs IVFFlat", show_lines=True)
    table.add_column("Característica", style="cyan", width=24)
    table.add_column("HNSW", style="green", width=26)
    table.add_column("IVFFlat", style="yellow", width=26)

    rows = [
        ("Algoritmo", "Grafo jerárquico", "Cuantización vectorial"),
        ("Query latency p50", "~2ms", "~8ms"),
        ("Build time", "Lento", "Rápido"),
        ("Memoria", "Alta (grafo en RAM)", "Baja"),
        ("Recall @10", "~99%", "~95% (varía con probes)"),
        ("Dataset óptimo", "< 5M vectores", "> 1M vectores"),
        ("Añadir nuevos docs", "Sin reconstruir", "Reconstruir si muchos"),
        ("Parámetro clave", "ef_search", "probes"),
        ("Disponible desde", "pgvector 0.5.0", "pgvector 0.4.0"),
        ("Recomendado para", "Producción general", "Datasets masivos"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)


def show_tuning():
    t1 = Table(title="Parámetros HNSW", show_lines=True)
    t1.add_column("Parámetro", style="cyan", width=18)
    t1.add_column("Default", width=10)
    t1.add_column("Rango", width=14)
    t1.add_column("Efecto al aumentar", style="white")

    for row in [
        ("m", "16", "4 – 64", "Más preciso, más RAM"),
        ("ef_construction", "64", "≥ 2 × m", "Mejor índice, más lento al indexar"),
        ("ef_search (SET)", "40", "1 – ∞", "Más preciso en búsqueda, más lento"),
    ]:
        t1.add_row(*row)
    console.print(t1)

    t2 = Table(title="Parámetros IVFFlat", show_lines=True)
    t2.add_column("Parámetro", style="cyan", width=18)
    t2.add_column("Fórmula", style="yellow", width=22)
    t2.add_column("Efecto", style="white")

    for row in [
        ("lists", "√N (ej: 10K docs → 100)", "Celdas del índice"),
        ("probes (SET)", "lists / 10", "Celdas inspeccionadas en búsqueda"),
    ]:
        t2.add_row(*row)
    console.print(t2)


def show_benchmark():
    console.print("\n[bold]Benchmark de referencia (10K documentos, k=4)[/bold]")

    table = Table(title="Latencia de búsqueda", show_lines=True)
    table.add_column("Configuración", style="cyan", width=30)
    table.add_column("Latencia p50", width=14)
    table.add_column("Latencia p99", width=14)
    table.add_column("Recall@10", width=12)

    rows = [
        ("Sin índice (sequential scan)", "~500ms", "~1200ms", "100%"),
        ("HNSW ef_search=40", "~2ms", "~5ms", "~97%"),
        ("HNSW ef_search=100", "~4ms", "~10ms", "~99%"),
        ("IVFFlat probes=10", "~8ms", "~20ms", "~95%"),
        ("IVFFlat probes=50", "~15ms", "~40ms", "~99%"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)
    console.print("[dim]Valores aproximados. Varían con hardware, dimensión de embedding y dataset.[/dim]")


def main():
    show_comparison()
    show_tuning()

    console.print(Panel(HNSW_SQL, title="HNSW Index (recomendado)", border_style="green"))
    console.print(Panel(IVFFLAT_SQL, title="IVFFlat Index (datasets masivos)", border_style="yellow"))
    console.print(Panel(METADATA_GIN_SQL, title="Índice en metadata JSONB", border_style="blue"))
    console.print(Panel(PARTIAL_SQL, title="Índice parcial por colección", border_style="dim"))

    show_benchmark()

    console.print(Panel(
        "1. Inserta datos PRIMERO, crea el índice DESPUÉS — es mucho más rápido\n"
        "2. Usa HNSW para producción general\n"
        "3. Ajusta ef_search según trade-off precisión / velocidad\n"
        "4. Añade índice GIN en cmetadata si filtras por metadata con frecuencia\n"
        "5. Verifica con EXPLAIN (ANALYZE, BUFFERS) que el índice se usa\n"
        "6. Los operadores de distancia deben coincidir: vector_cosine_ops + <=>",
        title="Mejores prácticas",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
