# Parent-Child Retriever — Precisión en búsqueda, contexto en respuesta

## El dilema del chunk size

El tamaño del chunk afecta dos cosas opuestas:

| Chunk pequeño (200 chars) | Chunk grande (1000 chars) |
|--------------------------|--------------------------|
| ✅ Búsqueda precisa | ❌ Búsqueda difusa (ruido) |
| ❌ Poco contexto para el LLM | ✅ Contexto amplio para el LLM |

**No puedes optimizar ambos con el mismo chunk.** La solución es usar dos tamaños distintos para cada propósito.

---

## Cómo funciona

```
Documento original
      │
      ├── parent_splitter (1000 chars)
      │         │
      │    [Padre 1][Padre 2][Padre 3]  → guardados en docstore
      │         │
      │    child_splitter (200 chars)
      │         │
      │    [h1][h2][h3][h4][h5]...      → vectorizados en ChromaDB
      │
Al buscar:
  query → ChromaDB → [h2, h4] más similares
                   → docstore → Padre 1, Padre 2  ← esto recibe el LLM
```

---

## Implementación

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,          # para los hijos (búsqueda)
    docstore=InMemoryStore(),         # para los padres (contexto)
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
)

# Indexar — crea hijos y padres automáticamente
retriever.add_documents(documentos)

# Buscar — retorna padres, aunque la búsqueda fue por hijos
docs = retriever.invoke("mi query")
```

---

## Variante: sin parent_splitter

Si omites `parent_splitter`, el "padre" es el documento completo:

```python
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
    # sin parent_splitter → el padre es el doc completo
)
```

Útil cuando los documentos son cortos (1-2 páginas) y quieres retornar el documento entero al LLM para que tenga máximo contexto.

---

## Docstore en producción

`InMemoryStore` se pierde al reiniciar. Para persistencia:

```python
# Redis
from langchain_community.storage import RedisStore
docstore = RedisStore(redis_url="redis://localhost:6379", key_prefix="parent:")

# Postgres / cualquier SQL
from langchain_community.storage import SQLStore
docstore = SQLStore(namespace="parents", db_url="postgresql://...")
```

El docstore solo necesita `get(keys)` y `mset(items)` — cualquier key-value funciona.

---

## Cuándo usar Parent-Child

| Situación | Usar |
|-----------|------|
| Documentos largos con secciones densas | ✅ |
| Queries precisos sobre detalles específicos | ✅ |
| Necesitas contexto amplio en la respuesta | ✅ |
| Documentos cortos (< 500 chars) | ❌ No tiene sentido |
| Prototipo rápido | ❌ Complejidad extra no justificada |

---

## Tradeoffs

| Ventaja | Contra |
|---------|--------|
| Búsqueda precisa + respuesta rica | Más complejidad (dos splitters, docstore extra) |
| Reduce alucinaciones por falta de contexto | El docstore necesita persistencia en producción |
| Flexible en tamaños | `add_documents()` recrea los hijos en cada carga |
