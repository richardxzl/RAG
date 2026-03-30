"""
25_metricas_manuales.py — Módulo 5.2: Métricas manuales

Implementa métricas de evaluación RAG sin dependencias externas:
  - Faithfulness: ¿la respuesta está soportada por el contexto recuperado?
  - Answer Relevance: ¿la respuesta responde a la pregunta?
  - Context Precision: ¿los chunks recuperados son relevantes?
  - Context Recall: ¿el ground truth está cubierto por los chunks?

Cada métrica usa el LLM como juez (LLM-as-a-judge pattern).
"""
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from rag.chain import get_llm, build_query_chain, format_docs
from rag.retriever import get_retriever

console = Console()
DATASET_PATH = Path("eval_dataset.json")


# ── Modelos Pydantic para los scores ─────────────────────────────────────────

class FaithfulnessScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Score de 0.0 a 1.0")
    razon: str = Field(description="Explicación del score en una oración")


class RelevanceScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Score de 0.0 a 1.0")
    razon: str = Field(description="Explicación del score en una oración")


# ── Prompts de evaluación (LLM-as-a-judge) ───────────────────────────────────

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un evaluador experto de sistemas RAG. Evalúa si la RESPUESTA está completamente "
     "soportada por el CONTEXTO proporcionado. "
     "Score 1.0 = totalmente soportada, 0.0 = inventada o contradice el contexto.\n\n"
     "{format_instructions}"),
    ("human",
     "CONTEXTO:\n{context}\n\nRESPUESTA:\n{answer}\n\nEvalúa la faithfulness:"),
])

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un evaluador experto de sistemas RAG. Evalúa si la RESPUESTA responde "
     "directamente a la PREGUNTA. "
     "Score 1.0 = responde perfectamente, 0.0 = irrelevante o no responde.\n\n"
     "{format_instructions}"),
    ("human",
     "PREGUNTA:\n{question}\n\nRESPUESTA:\n{answer}\n\nEvalúa la relevancia:"),
])


# ── Funciones de métricas ─────────────────────────────────────────────────────

def evaluar_faithfulness(context: str, answer: str, llm) -> FaithfulnessScore:
    """¿La respuesta está soportada por el contexto? No inventa nada."""
    parser = PydanticOutputParser(pydantic_object=FaithfulnessScore)
    chain = FAITHFULNESS_PROMPT | llm | parser
    return chain.invoke({
        "context": context[:2000],  # limitar para no exceder tokens
        "answer": answer,
        "format_instructions": parser.get_format_instructions(),
    })


def evaluar_relevance(question: str, answer: str, llm) -> RelevanceScore:
    """¿La respuesta responde a la pregunta que se hizo?"""
    parser = PydanticOutputParser(pydantic_object=RelevanceScore)
    chain = RELEVANCE_PROMPT | llm | parser
    return chain.invoke({
        "question": question,
        "answer": answer,
        "format_instructions": parser.get_format_instructions(),
    })


def evaluar_context_precision(question: str, docs: list, llm) -> float:
    """
    Proporción de chunks recuperados que son realmente relevantes para la pregunta.
    Usa el LLM para juzgar cada chunk individualmente.
    """
    if not docs:
        return 0.0

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responde solo 'sí' o 'no': ¿Este fragmento contiene información "
                   "relevante para responder la pregunta?"),
        ("human", "Pregunta: {question}\n\nFragmento:\n{chunk}"),
    ])
    from langchain_core.output_parsers import StrOutputParser
    chain = prompt | llm | StrOutputParser()

    relevantes = 0
    for doc in docs:
        respuesta = chain.invoke({
            "question": question,
            "chunk": doc.page_content[:500],
        }).strip().lower()
        if "sí" in respuesta or "si" in respuesta or "yes" in respuesta:
            relevantes += 1

    return relevantes / len(docs)


# ── Pipeline de evaluación completo ──────────────────────────────────────────

def evaluar_sample(sample: dict, llm, retriever) -> dict:
    """Evalúa un sample del dataset y retorna las métricas."""
    question = sample["question"]

    # Obtener respuesta del RAG
    docs = retriever.invoke(question)
    context = format_docs(docs)

    from langchain_core.prompts import ChatPromptTemplate as CPT
    from langchain_core.output_parsers import StrOutputParser
    from rag.chain import QUERY_PROMPT
    from langchain_core.runnables import RunnablePassthrough

    chain = (
        RunnablePassthrough.assign(context=lambda x: context)
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke({"question": question, "docs": docs})

    # Calcular métricas
    faith = evaluar_faithfulness(context, answer, llm)
    rel = evaluar_relevance(question, answer, llm)
    cp = evaluar_context_precision(question, docs, llm)

    return {
        "id": sample["id"],
        "question": question,
        "answer": answer[:120] + "...",
        "faithfulness": faith.score,
        "faith_razon": faith.razon,
        "answer_relevance": rel.score,
        "rel_razon": rel.razon,
        "context_precision": cp,
        "num_docs": len(docs),
    }


def main():
    console.rule("[bold blue]RAG Lab — Módulo 5.2: Métricas Manuales")

    if not DATASET_PATH.exists():
        console.print("[red]Ejecuta primero 24_eval_dataset.py para crear el dataset.[/]")
        return

    dataset = json.loads(DATASET_PATH.read_text())
    samples = dataset["samples"][:3]  # evaluar 3 para no gastar muchos tokens

    llm = get_llm()
    retriever = get_retriever()

    resultados = []
    for sample in samples:
        console.print(f"\n[dim]Evaluando: {sample['question'][:60]}...[/]")
        r = evaluar_sample(sample, llm, retriever)
        resultados.append(r)

    # Tabla de resultados
    table = Table(title="Métricas de evaluación", show_header=True, header_style="bold magenta")
    table.add_column("ID", width=6)
    table.add_column("Pregunta", max_width=30)
    table.add_column("Faithfulness", width=13)
    table.add_column("Relevance", width=10)
    table.add_column("Ctx Precision", width=13)

    for r in resultados:
        def color(v): return "green" if v >= 0.7 else "yellow" if v >= 0.4 else "red"
        table.add_row(
            r["id"],
            r["question"][:28] + "...",
            f"[{color(r['faithfulness'])}]{r['faithfulness']:.2f}[/]",
            f"[{color(r['answer_relevance'])}]{r['answer_relevance']:.2f}[/]",
            f"[{color(r['context_precision'])}]{r['context_precision']:.2f}[/]",
        )
    console.print(table)

    # Promedios
    if resultados:
        avg_f = sum(r["faithfulness"] for r in resultados) / len(resultados)
        avg_r = sum(r["answer_relevance"] for r in resultados) / len(resultados)
        avg_cp = sum(r["context_precision"] for r in resultados) / len(resultados)
        console.print(Panel(
            f"[bold]Promedios ({len(resultados)} samples):[/]\n"
            f"  Faithfulness:      {avg_f:.2f}\n"
            f"  Answer Relevance:  {avg_r:.2f}\n"
            f"  Context Precision: {avg_cp:.2f}",
            border_style="cyan",
        ))


if __name__ == "__main__":
    main()
