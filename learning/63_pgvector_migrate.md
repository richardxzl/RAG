# Migrar de ChromaDB a pgvector

## ¿Por qué migrar?

| ChromaDB | pgvector (Supabase) |
|---------|---------------------|
| Archivos locales | PostgreSQL en la nube |
| Sin auth / multitenancy | Row Level Security nativo |
| Filtros con DSL propio | SQL completo |
| Para dev / prototipo | Para producción real |
| Sin backup automático | Backup incluido en Supabase |

---

## Instalación

```bash
pip install langchain-postgres psycopg2-binary
```

En Supabase (SQL Editor):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## API idéntica a ChromaDB

```python
from langchain_postgres import PGVector

pgvector = PGVector(
    embeddings=get_embeddings(),
    collection_name="mi_knowledge_base",
    connection="postgresql://user:pass@host:5432/db",
)

# Misma API que Chroma — el código del RAG no cambia
pgvector.add_documents(docs)
docs = pgvector.similarity_search("query", k=4)
retriever = pgvector.as_retriever(search_kwargs={"k": 4})
```

Para migrar el proyecto: cambiar `get_vectorstore()` en `rag/retriever.py` para devolver `PGVector` en lugar de `Chroma`.

---

## Script de migración

```python
# 1. Leer de ChromaDB (con embeddings pre-calculados)
chroma_vs = get_vectorstore()
result = chroma_vs._collection.get(
    include=["documents", "metadatas", "embeddings"]
)

# 2. Insertar en pgvector reutilizando los embeddings
pgvector_vs.add_embeddings(
    texts=result["documents"],
    embeddings=result["embeddings"],  # no recalcula — mucho más rápido
    metadatas=result["metadatas"],
)
```

Si `embeddings` no está disponible, usar `add_documents(docs)` — recalcula pero siempre funciona.

---

## Tabla creada automáticamente

```sql
CREATE TABLE langchain_pg_embedding (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID,
    embedding     vector(384),   -- igual que EMBEDDING_DIMS
    document      TEXT,
    cmetadata     JSONB          -- metadata del chunk
);
```

---

## Connection string de Supabase

En Project Settings → Database → Connection string (Transaction mode para serverless):
```
postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
```
