# Metadata Enrichment — Agregar contexto a cada chunk

## Por qué importa la metadata

La metadata no solo es para mostrar fuentes — **habilita capacidades**:

| Metadata | Qué habilita |
|----------|-------------|
| `source`, `file_name` | Identificar el documento origen |
| `page` | Citar la página exacta |
| `chunk_index`, `total_chunks` | Saber si es el inicio, medio o fin del documento |
| `section` | Self-Query por sección (Módulo 3.6) |
| `file_type` | Filtrar por tipo de documento |
| `word_count` | Detectar chunks muy cortos o muy largos |
| `ingested_at` | Filtrar documentos recientes |
| `author`, `category` | Filtros semánticos en dominios específicos |

---

## Patrón de enriquecimiento

El enriquecimiento debe ocurrir en la **ingesta** (01_ingest.py), no en el retrieval. Una vez que los chunks están en el vector store, no puedes modificar su metadata fácilmente.

```python
def enrich_metadata(doc: Document, chunk_index: int, total_chunks: int) -> Document:
    source = doc.metadata.get("source", "")
    return Document(
        page_content=doc.page_content,
        metadata={
            **doc.metadata,                    # metadata original
            "file_name": os.path.basename(source),
            "file_type": os.path.splitext(source)[1].lstrip("."),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "char_count": len(doc.page_content),
            "word_count": len(doc.page_content.split()),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
    )
```

---

## Metadata vs contenido del chunk

Hay una decisión de diseño: ¿poner cierta información como metadata o incluirla en el texto del chunk?

```python
# Opción A: solo metadata (no visible para el LLM en la búsqueda semántica)
doc.metadata["author"] = "Juan García"

# Opción B: en el contenido (el LLM puede razonar sobre ello)
doc.page_content = f"Autor: Juan García\n\n{doc.page_content}"

# Opción C: ambas (filtrable Y visible para el LLM)
doc.metadata["author"] = "Juan García"
doc.page_content = f"[Autor: Juan García]\n\n{doc.page_content}"
```

**Regla**: si necesitas **filtrar** por ese campo (Self-Query, EmbeddingsFilter), ponlo en metadata. Si necesitas que el LLM lo **razone** (mencione al autor en la respuesta), ponlo también en el contenido.

---

## Integrar en el pipeline de ingesta

En `01_ingest.py`, después de hacer el split:

```python
# Antes
chunks = splitter.split_documents(docs)
vectorstore.add_documents(chunks)

# Después (con enriquecimiento)
chunks = splitter.split_documents(docs)
chunks_enriquecidos = enrich_document_chunks(chunks)  # función del módulo 4.3
vectorstore.add_documents(chunks_enriquecidos)
```

---

## Metadata para dominios específicos

| Dominio | Metadata útil |
|---------|--------------|
| Base de conocimiento técnica | versión, producto, módulo, última_actualización |
| Base legal / normativa | jurisdicción, fecha_vigencia, artículo, ley |
| Base de soporte | tipo_incidencia, severidad, estado, producto |
| Base académica | autor, año, revista, DOI, área |

Diseña la metadata según las preguntas que anticipas del usuario. Si el usuario preguntará "¿qué dice la versión 2.0?", necesitas metadata `version`.

---

## Limitaciones

- **ChromaDB**: soporta strings, integers, floats y booleans en metadata. No soporta listas como metadata filtrable (sí como dato, pero no para `where` queries).
- **Metadata estática**: una vez indexada, modificar la metadata requiere reindexar el chunk.
- **No abuses**: demasiada metadata aumenta el tiempo de indexación y el tamaño del store.
