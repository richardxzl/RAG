"""
17_multi_query_retriever.py — Módulo 3.4: Multi-Query Retriever

El LLM genera variaciones de la pregunta original para ampliar el recall.
Problema que resuelve: una sola formulación de la pregunta puede no recuperar
todos los chunks relevantes si el embedding no captura bien esa perspectiva.

Flujo: pregunta → LLM genera 3-5 variaciones → retrieval por cada una → deduplicar → contexto
"""
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.retriever import get_retriever
from rag.chain import get_llm, QUERY_PROMPT, format_docs

console = Console()

QUERIES = [
    "¿Qué tecnologías se mencionan?",
    "Resume los puntos principales",
]


def build_multi_query_retriever(verbose: bool = True):
    """
    MultiQueryRetriever usa el LLM para generar variaciones del query.
    Con logging activado, podemos ver qué variaciones generó.
    """
    retriever = get_retriever()
    llm = get_llm()

    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )

    if verbose:
        # Activar logging para ver las queries generadas
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    return mq_retriever


def comparar_retrievers(query: str):
    """Compara retriever normal vs multi-query para el mismo query."""
    console.print(f"\n[bold]Query original:[/] {query}")

    # Normal
    retriever_normal = get_retriever()
    docs_normal = retriever_normal.invoke(query)

    # Multi-query
    console.print("[dim]Generando variaciones del query...[/]")
    mq = MultiQueryRetriever.from_llm(retriever=get_retriever(), llm=get_llm())
    docs_mq = mq.invoke(query)

    # Comparación
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Retriever")
    table.add_column("Docs únicos recuperados")
    table.add_column("Primer chunk (preview)")

    preview_n = docs_normal[0].page_content[:70].replace("\n", " ") if docs_normal else "-"
    preview_mq = docs_mq[0].page_content[:70].replace("\n", " ") if docs_mq else "-"

    table.add_row("Normal", str(len(docs_normal)), preview_n)
    table.add_row("Multi-Query", str(len(docs_mq)), preview_mq)
    console.print(table)

    # Docs exclusivos del multi-query (aportados por las variaciones)
    textos_n = {d.page_content[:80] for d in docs_normal}
    textos_mq = {d.page_content[:80] for d in docs_mq}
    extras = textos_mq - textos_n

    if extras:
        console.print(f"  [cyan]+{len(extras)} docs adicionales aportados por las variaciones del query[/]")
    else:
        console.print("  [dim]No hubo docs adicionales (corpus pequeño o query muy preciso)[/]")


def demo_pipeline_completo(query: str):
    """Pipeline RAG completo usando MultiQueryRetriever."""
    console.rule("[bold yellow]Pipeline RAG con Multi-Query")

    llm = get_llm()
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=get_retriever(),
        llm=llm,
    )

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(mq_retriever.invoke(x["question"])),
        )
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )

    respuesta = chain.invoke({"question": query})
    console.print(Panel(respuesta, title=f"Respuesta para: {query}", border_style="green"))


def main():
    console.rule("[bold blue]RAG Lab — Módulo 3.4: Multi-Query Retriever")

    console.print(Panel(
        "MultiQueryRetriever usa el LLM para generar 3-5 variaciones del query original.\n"
        "Cada variación se ejecuta por separado en el retriever.\n"
        "Los resultados se fusionan eliminando duplicados.\n\n"
        "[dim]Ejemplo: 'ventajas de LCEL' puede generar:\n"
        "  · '¿Cuáles son los beneficios de LangChain Expression Language?'\n"
        "  · '¿Por qué usar LCEL en vez de chains legacy?'\n"
        "  · '¿Qué mejoras aporta LCEL al desarrollo de pipelines?'[/]",
        title="Concepto",
        border_style="blue",
    ))

    for q in QUERIES:
        comparar_retrievers(q)

    demo_pipeline_completo(QUERIES[0])


if __name__ == "__main__":
    main()
