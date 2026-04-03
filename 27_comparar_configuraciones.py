"""
27_comparar_configuraciones.py — Módulo 5.4: Comparar configuraciones

Grid search sobre: chunk_size × retriever_type × k
Evalúa cada combinación con las métricas del módulo 5.2 (LLM-as-a-judge).
Identifica la configuración óptima para el corpus actual.

Configuraciones comparadas:
  chunk_size: [500, 1000]
  retriever: [similarity, mmr]
  k: [3, 5]
"""
import json
import time
import logging
from pathlib import Path
from itertools import product
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_vectorstore
from rag.embeddings import get_embeddings

console = Console()
DATASET_PATH = Path("eval_dataset.json")

# Grid de configuraciones a comparar
CHUNK_SIZES = [500, 1000]
RETRIEVER_TYPES = ["similarity", "mmr"]
K_VALUES = [3, 5]


class EvalScore(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)


EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Evalúa la calidad de la respuesta RAG. "
     "faithfulness: ¿la respuesta está soportada por el contexto? (0-1). "
     "relevance: ¿la respuesta responde la pregunta? (0-1).\n\n"
     "{format_instructions}"),
    ("human",
     "PREGUNTA: {question}\nCONTEXTO: {context}\nRESPUESTA: {answer}"),
])


def build_temp_vectorstore(chunk_size: int):
    """Crea un vector store temporal re-chunkeando los docs del store principal."""
    vs_principal = get_vectorstore()
    data = vs_principal.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document
    docs_originales = [
        Document(page_content=c, metadata=m or {})
        for c, m in zip(data["documents"], data["metadatas"])
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2),
    )
    chunks = splitter.split_documents(docs_originales)

    vs_temp = Chroma(
        collection_name=f"grid_{chunk_size}",
        embedding_function=get_embeddings(),
    )
    if chunks:
        vs_temp.add_documents(chunks)
    return vs_temp


def build_retriever(vs, retriever_type: str, k: int):
    kwargs = {"k": k}
    if retriever_type == "mmr":
        kwargs["fetch_k"] = k * 4
        kwargs["lambda_mult"] = 0.5
    return vs.as_retriever(search_type=retriever_type, search_kwargs=kwargs)


def evaluar_muestra(question: str, retriever, llm, parser, eval_chain) -> dict:
    """Ejecuta el RAG y evalúa la respuesta para un query."""
    t0 = time.time()
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = (
        RunnablePassthrough.assign(context=lambda x: context)
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke({"question": question, "docs": docs})
    latencia = (time.time() - t0) * 1000

    try:
        score = eval_chain.invoke({
            "question": question,
            "context": context[:1500],
            "answer": answer,
            "format_instructions": parser.get_format_instructions(),
        })
        return {
            "faithfulness": score.faithfulness,
            "relevance": score.relevance,
            "latencia_ms": latencia,
            "num_docs": len(docs),
        }
    except Exception:
        return {"faithfulness": 0.0, "relevance": 0.0, "latencia_ms": latencia, "num_docs": len(docs)}


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 5.4: Comparar Configuraciones")

    if not DATASET_PATH.exists():
        console.print("[red]Ejecuta primero 24_eval_dataset.py[/]")
        return

    dataset = json.loads(DATASET_PATH.read_text())
    # Solo 2 preguntas para mantener el costo bajo
    samples = dataset["samples"][:2]
    questions = [s["question"] for s in samples]

    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=EvalScore)
    eval_chain = EVAL_PROMPT | llm | parser

    resultados = []
    combinaciones = list(product(CHUNK_SIZES, RETRIEVER_TYPES, K_VALUES))
    total = len(combinaciones)

    console.print(f"\n[dim]Evaluando {total} configuraciones × {len(questions)} preguntas...[/]\n")

    vectorstores = {}
    for chunk_size in CHUNK_SIZES:
        console.print(f"[dim]Construyendo vector store chunk_size={chunk_size}...[/]")
        vectorstores[chunk_size] = build_temp_vectorstore(chunk_size)

    for i, (chunk_size, ret_type, k) in enumerate(combinaciones, 1):
        config_id = f"cs{chunk_size}_{ret_type}_k{k}"
        console.print(f"[dim][{i}/{total}] {config_id}...[/]")

        vs = vectorstores[chunk_size]
        retriever = build_retriever(vs, ret_type, k)

        scores_faith = []
        scores_rel = []
        latencias = []

        for q in questions:
            r = evaluar_muestra(q, retriever, llm, parser, eval_chain)
            scores_faith.append(r["faithfulness"])
            scores_rel.append(r["relevance"])
            latencias.append(r["latencia_ms"])

        resultados.append({
            "config": config_id,
            "chunk_size": chunk_size,
            "retriever": ret_type,
            "k": k,
            "faithfulness": sum(scores_faith) / len(scores_faith),
            "relevance": sum(scores_rel) / len(scores_rel),
            "latencia_ms": sum(latencias) / len(latencias),
        })

    # Tabla de resultados
    table = Table(title="Grid Search — Resultados", show_header=True, header_style="bold magenta")
    table.add_column("Configuración")
    table.add_column("Chunk", width=6)
    table.add_column("Retriever", width=10)
    table.add_column("k", width=3)
    table.add_column("Faithfulness")
    table.add_column("Relevance")
    table.add_column("Latencia")

    def fmt(v):
        color = "green" if v >= 0.7 else "yellow" if v >= 0.4 else "red"
        return f"[{color}]{v:.2f}[/]"

    # Ordenar por promedio de métricas
    resultados.sort(key=lambda r: (r["faithfulness"] + r["relevance"]) / 2, reverse=True)

    for r in resultados:
        table.add_row(
            r["config"],
            str(r["chunk_size"]),
            r["retriever"],
            str(r["k"]),
            fmt(r["faithfulness"]),
            fmt(r["relevance"]),
            f"{r['latencia_ms']:.0f}ms",
        )
    console.print(table)

    mejor = resultados[0]
    console.print(Panel(
        f"[bold]Mejor configuración:[/] {mejor['config']}\n"
        f"  Faithfulness: {mejor['faithfulness']:.2f}\n"
        f"  Relevance:    {mejor['relevance']:.2f}\n"
        f"  Latencia:     {mejor['latencia_ms']:.0f}ms\n\n"
        f"[dim]Nota: con solo {len(questions)} preguntas, los resultados son orientativos.\n"
        f"Para conclusiones sólidas usa el dataset completo (módulo 5.1).[/]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
