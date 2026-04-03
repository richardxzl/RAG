"""
21_semantic_chunking.py — Módulo 4.2: Semantic Chunking

Divide el texto por cambios semánticos en vez de por tamaño fijo.
El splitter calcula la similitud entre oraciones consecutivas y corta
cuando el cambio de tema es suficientemente grande.

Requiere: pip install langchain-experimental
"""
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag.embeddings import get_embeddings

console = Console()

TEXTO_EJEMPLO = """
LangChain es un framework de código abierto diseñado para facilitar el desarrollo de aplicaciones
basadas en modelos de lenguaje. Proporciona herramientas para conectar LLMs con fuentes de datos
externas, APIs y otros sistemas. Su arquitectura modular permite combinar componentes de forma flexible.

Los embeddings son representaciones vectoriales del texto. Cada palabra o fragmento de texto se
convierte en un vector de números en un espacio de alta dimensión. La similitud entre textos se
mide como la distancia coseno entre sus vectores. Los modelos de embedding como all-MiniLM-L6-v2
capturan el significado semántico del texto de forma eficiente.

El fútbol es uno de los deportes más populares del mundo. El Real Madrid y el Barcelona son
los dos grandes clubes españoles. La Liga española es considerada una de las mejores del mundo.
Messi y Ronaldo dominaron el fútbol durante más de una década.

Los sistemas RAG (Retrieval-Augmented Generation) combinan la búsqueda de información con
la generación de texto. Primero recuperan documentos relevantes de una base de conocimiento,
luego los usan como contexto para que el LLM genere una respuesta fundamentada. Esto reduce
las alucinaciones y mejora la precisión de las respuestas.

Redis es una base de datos en memoria de código abierto. Se usa principalmente como cache,
broker de mensajes y almacén de estructuras de datos. Soporta strings, hashes, listas y sets.
Su velocidad lo hace ideal para aplicaciones que requieren baja latencia.
"""


def demo_recursive_splitter():
    """Splitter por tamaño fijo — puede cortar en medio de un párrafo sobre el mismo tema."""
    console.rule("[bold yellow]RecursiveCharacterTextSplitter (tamaño fijo)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    doc = Document(page_content=TEXTO_EJEMPLO)
    chunks = splitter.split_documents([doc])

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Chars", width=6)
    table.add_column("Contenido (preview)", max_width=70)

    for i, chunk in enumerate(chunks, 1):
        table.add_row(str(i), str(len(chunk.page_content)), chunk.page_content[:90].replace("\n", " "))

    console.print(table)
    console.print(f"  [dim]{len(chunks)} chunks, tamaño fijo — puede mezclar temas[/]")


def demo_semantic_chunker(breakpoint_type: str = "percentile"):
    """
    SemanticChunker calcula similitud entre oraciones consecutivas.
    Cuando la similitud cae por debajo del breakpoint, corta ahí.

    breakpoint_threshold_type:
      - "percentile": corta donde la diferencia supera el percentil N
      - "standard_deviation": corta donde supera N desviaciones estándar
      - "interquartile": usa el rango intercuartílico
    """
    console.rule(f"[bold yellow]SemanticChunker (breakpoint_type={breakpoint_type})")

    splitter = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type=breakpoint_type,
        breakpoint_threshold_amount=95,  # percentil 95 → solo cortes muy marcados
    )

    chunks = splitter.create_documents([TEXTO_EJEMPLO])

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Chars", width=6)
    table.add_column("Contenido (preview)", max_width=70)

    for i, chunk in enumerate(chunks, 1):
        table.add_row(str(i), str(len(chunk.page_content)), chunk.page_content[:90].replace("\n", " "))

    console.print(table)
    console.print(f"  [dim]{len(chunks)} chunks — cortados por cambio semántico[/]")

    # ¿Separó el fútbol del resto de LangChain/embeddings/RAG?
    for i, chunk in enumerate(chunks):
        if "fútbol" in chunk.page_content.lower():
            console.print(f"  [cyan]Chunk {i+1} contiene el párrafo de fútbol — "
                          f"{'aislado' if len(chunks) > 2 else 'mezclado'}[/]")

    return chunks


def comparar_coherencia(chunks_fixed: list, chunks_semantic: list):
    """Compara la coherencia temática de ambos enfoques."""
    console.rule("[bold yellow]Comparativa de coherencia")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Métrica")
    table.add_column("Tamaño fijo")
    table.add_column("Semántico")

    avg_fixed = sum(len(c.page_content) for c in chunks_fixed) // max(len(chunks_fixed), 1)
    avg_semantic = sum(len(c.page_content) for c in chunks_semantic) // max(len(chunks_semantic), 1)

    table.add_row("Total chunks", str(len(chunks_fixed)), str(len(chunks_semantic)))
    table.add_row("Longitud promedio", str(avg_fixed), str(avg_semantic))
    table.add_row(
        "Respeta cambios de tema",
        "[red]No[/]",
        "[green]Sí[/]",
    )
    table.add_row(
        "Tamaño predecible",
        "[green]Sí[/]",
        "[yellow]Variable[/]",
    )
    table.add_row(
        "Requiere embeddings para indexar",
        "[green]No[/]",
        "[yellow]Sí (2× más lento)[/]",
    )

    console.print(table)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 4.2: Semantic Chunking")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    splitter_fixed = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks_fixed = splitter_fixed.split_documents([Document(page_content=TEXTO_EJEMPLO)])

    demo_recursive_splitter()
    chunks_semantic = demo_semantic_chunker("percentile")
    comparar_coherencia(chunks_fixed, chunks_semantic)

    console.print(Panel(
        "[bold]Cuándo usar SemanticChunker:[/]\n\n"
        "✅ Documentos con secciones temáticamente distintas sin headers claros\n"
        "✅ Texto narrativo (artículos, ensayos, transcripciones)\n"
        "✅ Cuando el tamaño variable es aceptable\n\n"
        "❌ Documentos técnicos estructurados (usar MarkdownHeaderTextSplitter)\n"
        "❌ Cuando la velocidad de ingesta es crítica (2× más lento)\n"
        "❌ Documentos muy homogéneos (el splitter no encuentra cambios)[/]",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
