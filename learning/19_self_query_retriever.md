# Self-Query Retriever — El LLM genera filtros de metadata

## El problema

Los usuarios formulan queries en lenguaje natural que implícitamente contienen filtros:

```
"Documentos de Python publicados en 2024"
→ semántico: "Python"
→ filtro: año = 2024

"Errores en la página 5 del manual"
→ semántico: "errores"
→ filtro: page = 5
```

Sin Self-Query, una búsqueda vectorial trata todo el query como semántica — el filtro `año = 2024` nunca se aplica como condición.

---

## Cómo funciona

```
Query: "documentos sobre RAG de la primera sección"
      │
      ▼
  LLM analiza el query
      │
      ├── query semántico: "RAG"
      └── filtro: {"page": {"$lte": 2}}
      │
      ▼
  ChromaDB:
    - búsqueda vectorial con "RAG"
    - filtro WHERE page <= 2
      │
      ▼
  Resultados precisos
```

---

## Implementación

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# 1. Definir qué metadata pueden filtrar
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="Nombre del archivo fuente",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="Número de página",
        type="integer",
    ),
    AttributeInfo(
        name="author",
        description="Autor del documento",
        type="string",
    ),
]

# 2. Crear el retriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Documentos técnicos sobre LangChain y RAG",
    metadata_field_info=metadata_field_info,
    verbose=True,  # muestra el query estructurado generado
)

# 3. Invocar — igual que cualquier retriever
docs = retriever.invoke("¿Qué dice el capítulo 3 sobre embeddings?")
```

---

## Operadores de filtro disponibles

| Operador | Significado |
|----------|-------------|
| `$eq` | igual |
| `$ne` | distinto |
| `$lt`, `$lte` | menor, menor o igual |
| `$gt`, `$gte` | mayor, mayor o igual |
| `$in` | en lista |
| `$nin` | no en lista |
| `$and`, `$or` | combinación lógica |

Ejemplo de filtro compuesto:
```
"Documentos de Python o JavaScript del año 2023"
→ {"$and": [{"year": {"$eq": 2023}}, {"$or": [{"topic": "python"}, {"topic": "javascript"}]}]}
```

---

## Cuándo vale la pena

Self-Query solo aporta valor real si los documentos tienen **metadata rica**. Con solo `source` y `page`, el LLM raramente puede construir filtros útiles.

| Metadata disponible | Valor de Self-Query |
|--------------------|---------------------|
| Solo source, page | Bajo |
| + fecha, autor, categoría | Medio |
| + tema, idioma, nivel, versión | Alto |

**Regla**: antes de implementar Self-Query, implementa primero Metadata Enrichment (Módulo 4.3).

---

## Compatibilidad con vector stores

Self-Query requiere que el vector store soporte filtrado por metadata. ChromaDB, Pinecone, Weaviate, y pgvector lo soportan. Algunos stores tienen sintaxis de filtro ligeramente diferente — LangChain lo abstrae, pero puede haber limitaciones.

---

## Tradeoffs

| Ventaja | Contra |
|---------|--------|
| Queries más precisos sin conocer la estructura | +1 llamada LLM para generar el query estructurado |
| El usuario no necesita saber el nombre de los campos | Depende fuertemente de la calidad de la metadata |
| Combina semántica + filtros exactos | El LLM puede malinterpretar filtros implícitos |
