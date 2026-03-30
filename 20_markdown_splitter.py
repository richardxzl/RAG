"""
20_markdown_splitter.py — Módulo 4.1: Markdown Splitter

Demuestra cómo respetar la estructura de documentos Markdown al dividir en chunks.
MarkdownHeaderTextSplitter corta por headers (H1, H2, H3) en vez de por caracteres.

Problema con RecursiveCharacterTextSplitter en Markdown:
  - Puede cortar a mitad de una sección
  - No respeta la jerarquía de headers
  - Pierde el contexto de "en qué sección estaba este chunk"
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

console = Console()

# Documento Markdown de ejemplo
MARKDOWN_EJEMPLO = """# LangChain — Guía de inicio

LangChain es un framework para construir aplicaciones con LLMs.

## Instalación

Para instalar LangChain ejecuta:

```bash
pip install langchain langchain-core
```

También necesitas una clave de API para el LLM.

## LCEL — LangChain Expression Language

LCEL es el sistema de composición moderno. Usa el operador pipe `|` para conectar componentes.

### Ejemplo básico

```python
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"question": "Hola"})
```

### Ventajas de LCEL

- Streaming nativo
- Batch nativo
- Composabilidad total

## Retrievers

Los retrievers buscan documentos relevantes en el vector store.

### Tipos de retriever

Hay varios tipos: similarity, MMR, multi-query, parent-child.

### Cuándo usar cada uno

Depende del caso de uso y del corpus.

## Conclusión

LangChain facilita la construcción de aplicaciones RAG y agentes.
"""


def demo_recursive_splitter():
    """El splitter por defecto no respeta la estructura Markdown."""
    console.rule("[bold yellow]RecursiveCharacterTextSplitter (sin respetar headers)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    doc = Document(page_content=MARKDOWN_EJEMPLO)
    chunks = splitter.split_documents([doc])

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Chars", width=6)
    table.add_column("Metadata")
    table.add_column("Contenido (preview)", max_width=60)

    for i, chunk in enumerate(chunks, 1):
        table.add_row(
            str(i),
            str(len(chunk.page_content)),
            str(chunk.metadata),
            chunk.page_content[:80].replace("\n", " "),
        )
    console.print(table)
    console.print(f"  [dim]Total chunks: {len(chunks)} — metadata vacía, sin contexto de sección[/]")


def demo_markdown_splitter():
    """MarkdownHeaderTextSplitter preserva la jerarquía y la agrega como metadata."""
    console.rule("[bold yellow]MarkdownHeaderTextSplitter (respeta headers)")

    # Definir qué headers cortan y cómo se llama la metadata
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,   # False = mantener el header en el contenido del chunk
    )

    chunks = splitter.split_text(MARKDOWN_EJEMPLO)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Chars", width=6)
    table.add_column("Metadata (sección)", max_width=30)
    table.add_column("Contenido (preview)", max_width=50)

    for i, chunk in enumerate(chunks, 1):
        meta_str = " > ".join(
            v for k, v in sorted(chunk.metadata.items())
            if v
        )
        table.add_row(
            str(i),
            str(len(chunk.page_content)),
            meta_str,
            chunk.page_content[:80].replace("\n", " "),
        )
    console.print(table)
    console.print(f"  [dim]Total chunks: {len(chunks)} — cada chunk sabe en qué sección está[/]")

    return chunks


def demo_pipeline_completo(chunks_md):
    """
    Después del MarkdownHeaderTextSplitter, aplicar RecursiveCharacterTextSplitter
    para asegurar que los chunks no excedan el tamaño máximo.
    """
    console.rule("[bold yellow]Pipeline completo: MarkdownHeader + RecursiveCharacter")

    # Segundo split: si algún chunk es muy largo, dividirlo respetando el tamaño máximo
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
    )

    # split_documents preserva la metadata del chunk padre
    chunks_finales = secondary_splitter.split_documents(chunks_md)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Chars", width=6)
    table.add_column("Metadata preservada", max_width=30)
    table.add_column("Contenido (preview)", max_width=50)

    for i, chunk in enumerate(chunks_finales, 1):
        meta_str = " > ".join(
            v for k, v in sorted(chunk.metadata.items()) if v
        )
        table.add_row(
            str(i),
            str(len(chunk.page_content)),
            meta_str,
            chunk.page_content[:70].replace("\n", " "),
        )
    console.print(table)
    console.print(
        f"  [dim]{len(chunks_md)} chunks → {len(chunks_finales)} chunks (después del segundo split)[/]"
    )


def main():
    console.rule("[bold blue]RAG Lab — Módulo 4.1: Markdown Splitter")
    demo_recursive_splitter()
    chunks_md = demo_markdown_splitter()
    demo_pipeline_completo(chunks_md)

    console.print(Panel(
        "[bold]Patrón recomendado para documentos Markdown:[/]\n\n"
        "1. MarkdownHeaderTextSplitter → chunks por sección con metadata de headers\n"
        "2. RecursiveCharacterTextSplitter → subdividir chunks muy largos\n"
        "3. La metadata de headers se preserva → el retriever puede filtrar por sección\n\n"
        "[dim]Esto conecta directamente con Self-Query Retriever (Módulo 3.6):\n"
        "con 'h1' y 'h2' en la metadata, el LLM puede filtrar por sección.[/]",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
