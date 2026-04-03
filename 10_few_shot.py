"""
10_few_shot.py — Módulo 2.2: Few-shot prompting

Demuestra cómo enseñar al LLM con ejemplos dentro del prompt:
  - Zero-shot: sin ejemplos (línea base)
  - Few-shot estático: ejemplos fijos con FewShotChatMessagePromptTemplate
  - Few-shot dinámico: selección automática con SemanticSimilarityExampleSelector

Caso de uso: clasificador de sentimiento de reseñas de productos.
"""
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

from rag.chain import get_llm
from rag.embeddings import get_embeddings

console = Console()

# ── Ejemplos para few-shot ────────────────────────────────────────────────────

EJEMPLOS = [
    {"resena": "El producto llegó en perfectas condiciones y superó mis expectativas.", "sentimiento": "positivo"},
    {"resena": "Pésima calidad, se rompió al segundo día de uso.", "sentimiento": "negativo"},
    {"resena": "Es lo que esperaba, ni más ni menos.", "sentimiento": "neutro"},
    {"resena": "Increíble relación calidad-precio, lo recomiendo totalmente.", "sentimiento": "positivo"},
    {"resena": "El envío tardó más de lo indicado y el embalaje estaba dañado.", "sentimiento": "negativo"},
    {"resena": "Funciona correctamente según las especificaciones.", "sentimiento": "neutro"},
    {"resena": "Mejor compra que he hecho en años, estoy encantado.", "sentimiento": "positivo"},
    {"resena": "No cumple con lo que promete la descripción, devuelto.", "sentimiento": "negativo"},
    {"resena": "Producto estándar, hace lo que tiene que hacer.", "sentimiento": "neutro"},
    {"resena": "La atención al cliente fue excelente cuando tuve un problema.", "sentimiento": "positivo"},
]

RESENAS_TEST = [
    "El color es diferente al de la foto pero la calidad es buena.",
    "¡Me encanta! Lo uso todos los días y funciona perfecto.",
    "Tardó 3 semanas en llegar y venía sin instrucciones.",
]


# ── Parte 1: Zero-shot (sin ejemplos) ────────────────────────────────────────

def demo_zero_shot(resena: str) -> str:
    llm = get_llm()
    template = ChatPromptTemplate.from_messages([
        ("system", "Clasifica el sentimiento de la reseña. "
                   "Responde SOLO con una palabra: positivo, negativo o neutro."),
        ("human", "{resena}"),
    ])
    chain = template | llm | StrOutputParser()
    return chain.invoke({"resena": resena}).strip().lower()


# ── Parte 2: Few-shot estático ────────────────────────────────────────────────

def demo_few_shot_estatico(resena: str) -> str:
    """
    FewShotChatMessagePromptTemplate inserta los ejemplos como pares
    human/ai antes de la pregunta real. El LLM infiere el patrón.
    """
    llm = get_llm()

    # Template para CADA ejemplo (cómo se formatea un par input/output)
    ejemplo_template = ChatPromptTemplate.from_messages([
        ("human", "{resena}"),
        ("ai", "{sentimiento}"),
    ])

    # Template few-shot con los 4 primeros ejemplos (estático)
    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=ejemplo_template,
        examples=EJEMPLOS[:4],
    )

    # Template completo: system + ejemplos + pregunta real
    template_final = ChatPromptTemplate.from_messages([
        ("system", "Clasifica el sentimiento de la reseña. "
                   "Responde SOLO con una palabra: positivo, negativo o neutro."),
        few_shot_template,         # ← aquí se expanden los ejemplos
        ("human", "{resena}"),
    ])

    chain = template_final | llm | StrOutputParser()
    return chain.invoke({"resena": resena}).strip().lower()


# ── Parte 3: Few-shot dinámico ────────────────────────────────────────────────

def build_selector_dinamico():
    """
    SemanticSimilarityExampleSelector selecciona los K ejemplos más
    similares semánticamente a la pregunta actual.
    Útil cuando tienes muchos ejemplos y no quieres incluirlos todos.
    """
    return SemanticSimilarityExampleSelector.from_examples(
        examples=EJEMPLOS,
        embeddings=get_embeddings(),
        vectorstore_cls=Chroma,
        k=3,                    # selecciona los 3 más similares
    )


def demo_few_shot_dinamico(resena: str, selector) -> tuple[str, list]:
    llm = get_llm()

    ejemplo_template = ChatPromptTemplate.from_messages([
        ("human", "{resena}"),
        ("ai", "{sentimiento}"),
    ])

    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=ejemplo_template,
        example_selector=selector,   # ← selector dinámico en vez de lista fija
    )

    template_final = ChatPromptTemplate.from_messages([
        ("system", "Clasifica el sentimiento de la reseña. "
                   "Responde SOLO con una palabra: positivo, negativo o neutro."),
        few_shot_template,
        ("human", "{resena}"),
    ])

    # Ver qué ejemplos seleccionó
    ejemplos_seleccionados = selector.select_examples({"resena": resena})

    chain = template_final | llm | StrOutputParser()
    resultado = chain.invoke({"resena": resena}).strip().lower()
    return resultado, ejemplos_seleccionados


# ── Demo completo ─────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 2.2: Few-shot Prompting")

    console.print("\n[dim]Construyendo selector semántico (vectoriza los ejemplos)...[/]")
    selector = build_selector_dinamico()

    resultados = []

    for resena in RESENAS_TEST:
        console.print(f"\n[bold]Reseña:[/] {resena}")

        r_zero    = demo_zero_shot(resena)
        r_estatico = demo_few_shot_estatico(resena)
        r_dinamico, ejemplos_usados = demo_few_shot_dinamico(resena, selector)

        console.print(f"  Zero-shot:       [cyan]{r_zero}[/]")
        console.print(f"  Few-shot fijo:   [cyan]{r_estatico}[/]")
        console.print(f"  Few-shot dinám.: [cyan]{r_dinamico}[/]")
        console.print("  [dim]Ejemplos seleccionados dinámicamente:[/]")
        for ej in ejemplos_usados:
            console.print(f"    · \"{ej['resena'][:50]}...\" → {ej['sentimiento']}")

        resultados.append((resena[:45] + "...", r_zero, r_estatico, r_dinamico))

    # Tabla resumen
    console.print()
    table = Table(title="Resumen comparativo", show_header=True, header_style="bold magenta")
    table.add_column("Reseña", style="dim", max_width=45)
    table.add_column("Zero-shot")
    table.add_column("Few-shot fijo")
    table.add_column("Few-shot dinámico")

    for resena, z, e, d in resultados:
        table.add_row(resena, z, e, d)

    console.print(table)


if __name__ == "__main__":
    main()
