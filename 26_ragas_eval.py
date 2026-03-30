"""
26_ragas_eval.py — Módulo 5.3: RAGAS — Evaluación automatizada end-to-end

RAGAS (Retrieval Augmented Generation Assessment) evalúa el pipeline RAG completo
con métricas estandarizadas sin necesidad de ground truth para todas las métricas.

Requiere: pip install ragas

Métricas que usa RAGAS:
  - faithfulness: ¿la respuesta está soportada por el contexto?
  - answer_relevancy: ¿la respuesta es relevante a la pregunta?
  - context_precision: ¿los chunks recuperados son precisos?
  - context_recall: ¿el contexto cubre el ground truth? (requiere ground truth)
"""
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
DATASET_PATH = Path("eval_dataset.json")


def check_ragas_installed() -> bool:
    try:
        import ragas
        return True
    except ImportError:
        return False


def preparar_dataset_ragas(samples: list[dict], retriever, llm) -> list[dict]:
    """
    RAGAS necesita un dataset con: question, answer, contexts, ground_truth.
    Lo construimos ejecutando el RAG sobre cada sample del dataset.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from rag.chain import QUERY_PROMPT, format_docs

    resultados = []
    for sample in samples:
        question = sample["question"]
        docs = retriever.invoke(question)
        context = format_docs(docs)

        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | QUERY_PROMPT
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke({"question": question, "docs": docs})

        resultados.append({
            "question": question,
            "answer": answer,
            "contexts": [d.page_content for d in docs],
            "ground_truth": sample.get("ground_truth", ""),
        })

    return resultados


def evaluar_con_ragas(dataset_ragas: list[dict]):
    """Ejecuta la evaluación RAGAS y retorna el resultado."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    hf_dataset = Dataset.from_list(dataset_ragas)

    resultado = evaluate(
        dataset=hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    return resultado


def mostrar_resultado_ragas(resultado):
    """Muestra los scores RAGAS en una tabla."""
    scores = resultado.to_pandas()

    table = Table(title="Evaluación RAGAS", show_header=True, header_style="bold magenta")
    table.add_column("Pregunta", max_width=35)
    table.add_column("Faithfulness")
    table.add_column("Ans. Relevancy")
    table.add_column("Ctx Precision")
    table.add_column("Ctx Recall")

    def fmt(v):
        if v != v:  # NaN check
            return "[dim]N/A[/]"
        color = "green" if v >= 0.7 else "yellow" if v >= 0.4 else "red"
        return f"[{color}]{v:.2f}[/]"

    for _, row in scores.iterrows():
        table.add_row(
            str(row.get("question", ""))[:33] + "...",
            fmt(row.get("faithfulness", float("nan"))),
            fmt(row.get("answer_relevancy", float("nan"))),
            fmt(row.get("context_precision", float("nan"))),
            fmt(row.get("context_recall", float("nan"))),
        )

    console.print(table)

    # Promedios
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    console.print("\n[bold]Promedios:[/]")
    for m in metrics:
        if m in scores.columns:
            avg = scores[m].mean()
            if avg == avg:  # not NaN
                color = "green" if avg >= 0.7 else "yellow" if avg >= 0.4 else "red"
                console.print(f"  {m:25} [{color}]{avg:.3f}[/]")


def demo_sin_ragas(samples: list[dict]):
    """Muestra el dataset que se pasaría a RAGAS sin ejecutarlo."""
    console.print("\n[bold]Dataset preparado para RAGAS (preview):[/]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Campo")
    table.add_column("Valor (preview)")
    table.add_column("Tipo")

    campos = [
        ("question", "¿Qué es LCEL?", "str"),
        ("answer", "LCEL es el sistema de composición...", "str (generado por el RAG)"),
        ("contexts", '["LCEL usa el operador |...", "..."]', "list[str] (chunks recuperados)"),
        ("ground_truth", "LCEL (LangChain Expression Language)...", "str (del dataset 5.1)"),
    ]
    for nombre, valor, tipo in campos:
        table.add_row(nombre, valor, tipo)
    console.print(table)


def main():
    console.rule("[bold blue]RAG Lab — Módulo 5.3: RAGAS")

    if not DATASET_PATH.exists():
        console.print("[red]Ejecuta primero 24_eval_dataset.py[/]")
        return

    dataset = json.loads(DATASET_PATH.read_text())
    samples = dataset["samples"][:3]

    if not check_ragas_installed():
        console.print(Panel(
            "[yellow]RAGAS no está instalado.[/]\n\n"
            "Para instalarlo:\n"
            "  [bold]pip install ragas datasets[/]\n\n"
            "Mientras tanto, mostramos la estructura del dataset que usaría RAGAS:",
            border_style="yellow",
        ))
        demo_sin_ragas(samples)
        console.print(Panel(
            "[bold]Cómo funciona RAGAS:[/]\n\n"
            "1. Prepara el dataset: {question, answer, contexts, ground_truth}\n"
            "2. evaluate(dataset, metrics=[faithfulness, answer_relevancy, ...])\n"
            "3. Retorna scores por sample + promedios\n\n"
            "RAGAS usa el LLM configurado en el entorno (OpenAI por defecto).\n"
            "Para usar Claude: configurar el LLM wrapper de RAGAS con langchain_anthropic.",
            border_style="blue",
        ))
        return

    # Si RAGAS está instalado, ejecutar la evaluación real
    from rag.chain import get_llm
    from rag.retriever import get_retriever

    llm = get_llm()
    retriever = get_retriever()

    console.print("[dim]Preparando dataset RAGAS...[/]")
    dataset_ragas = preparar_dataset_ragas(samples, retriever, llm)

    console.print("[dim]Ejecutando evaluación RAGAS...[/]")
    resultado = evaluar_con_ragas(dataset_ragas)
    mostrar_resultado_ragas(resultado)


if __name__ == "__main__":
    main()
