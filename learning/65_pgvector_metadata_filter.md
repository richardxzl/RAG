# Filtrado por Metadata en SQL

## La ventaja clave de pgvector

ChromaDB tiene su propio DSL de filtros. pgvector usa SQL — más potente, más flexible, y que ya conoces.

---

## Filtros con la API de LangChain

```python
# Igualdad
retriever = pgvector.as_retriever(
    search_kwargs={"k": 4, "filter": {"source": "doc.pdf"}}
)

# Operadores de comparación
filter = {
    "chunk_index": {"$gte": 5, "$lte": 20},
    "file_type": {"$in": ["pdf", "md"]},
}

# OR entre fuentes
filter = {
    "$or": [
        {"source": "doc1.pdf"},
        {"source": "doc2.pdf"},
    ]
}
```

Operadores disponibles: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`.

---

## SQL directo — máximo control

```sql
-- Full-text + vectorial combinados (hybrid search en SQL puro)
SELECT document, cmetadata,
       (1 - (embedding <=> '[vector]'::vector)) * 0.7
       + ts_rank(to_tsvector('spanish', document),
                 plainto_tsquery('spanish', 'LangChain')) * 0.3 AS score
FROM langchain_pg_embedding
WHERE collection_id = $1
ORDER BY score DESC LIMIT 4;

-- Deduplicar por fuente (evitar que un doc monopolice los resultados)
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
SELECT * FROM ranked WHERE rn = 1 ORDER BY dist LIMIT 4;
```

---

## Row Level Security (multi-tenant)

```sql
-- Activar RLS
ALTER TABLE langchain_pg_embedding ENABLE ROW LEVEL SECURITY;

-- Cada usuario solo ve sus documentos
CREATE POLICY user_isolation ON langchain_pg_embedding
    USING (cmetadata->>'user_id' = current_user);
```

Con RLS, el mismo RAG puede servir a múltiples usuarios sin cambiar nada en el código Python — la base de datos filtra automáticamente.

---

## Comparación de capacidades

| Capacidad | ChromaDB | pgvector |
|---------|----------|---------|
| Igualdad / comparación | ✓ | ✓ |
| Full-text search | ✗ | ✓ (ts_vector) |
| JOINs con otras tablas | ✗ | ✓ |
| Deduplicación (window fn) | ✗ | ✓ |
| Row Level Security | ✗ | ✓ nativo |
| Arrays en metadata | ✗ | ✓ (? operator) |
| Regex | ✗ | ✓ (~ operator) |
| Subqueries | ✗ | ✓ |

---

## Cuándo migrar a pgvector

- Necesitas multi-tenancy (cada usuario ve solo sus documentos)
- Quieres combinar búsqueda vectorial con full-text en una sola query
- Tienes que hacer JOINs con tablas de tu aplicación
- Necesitas auditoría o Row Level Security
- Quieres deduplicar resultados por fuente
