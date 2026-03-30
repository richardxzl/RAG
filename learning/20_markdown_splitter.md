# Markdown Splitter — Respetar la estructura del documento

## El problema con el splitter genérico

`RecursiveCharacterTextSplitter` no sabe nada sobre la estructura del documento. Divide por caracteres y puede cortar a mitad de una sección, mezclar contenido de secciones distintas, y no agrega ninguna metadata sobre el contexto:

```
Chunk 1: "...ventajas de LCEL\n- Streaming nativo\n- Batch nati"
Chunk 2: "vo\n- Composabilidad\n\n## Retrievers\nLos retrievers..."
```

Ese chunk 1 no sabe que pertenece a la sección "LCEL > Ventajas". El chunk 2 mezcla el final de una sección con el inicio de otra.

---

## MarkdownHeaderTextSplitter

Divide el documento en el límite de los headers, y agrega el path de headers como metadata:

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),    # H1 → metadata key "h1"
    ("##", "h2"),   # H2 → metadata key "h2"
    ("###", "h3"),  # H3 → metadata key "h3"
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # True = quitar el header del contenido del chunk
)

chunks = splitter.split_text(markdown_text)
# chunks[0].metadata = {"h1": "LangChain — Guía", "h2": "Instalación"}
# chunks[0].page_content = "## Instalación\nPara instalar..."
```

---

## Metadata de sección — el valor real

Cada chunk sabe exactamente en qué sección del documento está:

```python
chunk.metadata = {
    "h1": "LangChain — Guía de inicio",
    "h2": "LCEL — LangChain Expression Language",
    "h3": "Ventajas de LCEL",
}
```

Esto permite:
1. **Self-Query Retriever** (Módulo 3.6): el usuario puede pedir "contenido de la sección de instalación"
2. **Fuentes más precisas**: mostrar al usuario `Manual > LCEL > Ventajas` en vez de solo `manual.md, pág 3`
3. **Filtrado por metadata**: recuperar solo chunks de una sección específica

---

## Pipeline recomendado

`MarkdownHeaderTextSplitter` puede generar chunks muy largos si una sección es extensa. El patrón correcto es aplicar un segundo splitter:

```python
# Paso 1: dividir por headers
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks_por_seccion = md_splitter.split_text(texto)

# Paso 2: subdividir los que son demasiado largos
char_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks_finales = char_splitter.split_documents(chunks_por_seccion)
# La metadata de headers se preserva automáticamente en split_documents()
```

---

## strip_headers

```python
# strip_headers=True (default)
chunk.page_content = "Para instalar ejecuta pip install..."
# El header "## Instalación" no aparece en el contenido

# strip_headers=False
chunk.page_content = "## Instalación\nPara instalar ejecuta..."
# El header está en el contenido — más contexto para el LLM
```

Recomendación: `strip_headers=False` en la mayoría de casos. El header en el contenido ayuda al LLM a contextualizar la información.

---

## Otros splitters especializados

| Splitter | Para |
|----------|------|
| `MarkdownHeaderTextSplitter` | Documentos `.md` con headers |
| `HTMLHeaderTextSplitter` | Páginas HTML respetando `<h1>`, `<h2>` |
| `PythonCodeTextSplitter` | Código Python — corta por función/clase |
| `LatexTextSplitter` | Documentos LaTeX — corta por sección |
| `RecursiveCharacterTextSplitter` | Fallback genérico para cualquier texto |

---

## Tradeoffs

| Ventaja | Contra |
|---------|--------|
| Metadata de sección gratis | Solo sirve para docs con headers |
| No corta a mitad de sección | Secciones largas requieren segundo split |
| Chunks coherentes temáticamente | Secciones muy cortas generan chunks tiny |
