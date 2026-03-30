"""
28_regression_testing.py — Módulo 5.5: Regression Testing

Detecta si un cambio en el pipeline empeoró la calidad de las respuestas.
Compara los scores actuales contra un baseline guardado.

Flujo:
  1. Primera ejecución: genera y guarda el baseline
  2. Cambias algo en el pipeline
  3. Segunda ejecución: compara contra el baseline
  4. Si la diferencia supera el threshold, reporta regresión
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()
DATASET_PATH = Path("eval_dataset.json")
BASELINE_PATH = Path("eval_baseline.json")
REGRESSION_THRESHOLD = 0.05  # si cae más de 5%, es una regresión


class EvalScore(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)


EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Evalúa la calidad RAG. faithfulness: ¿respuesta soportada por contexto? "
     "relevance: ¿responde la pregunta? Cada score 0-1.\n\n{format_instructions}"),
    ("human", "PREGUNTA: {question}\nCONTEXTO: {context}\nRESPUESTA: {answer}"),
])


def run_eval(samples: list[dict], llm, retriever) -> list[dict]:
    """Ejecuta el pipeline RAG + evaluación sobre los samples."""
    parser = PydanticOutputParser(pydantic_object=EvalScore)
    eval_chain = EVAL_PROMPT | llm | parser

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

        try:
            score = eval_chain.invoke({
                "question": question,
                "context": context[:1500],
                "answer": answer,
                "format_instructions": parser.get_format_instructions(),
            })
            resultados.append({
                "id": sample["id"],
                "question": question,
                "faithfulness": score.faithfulness,
                "relevance": score.relevance,
            })
        except Exception as e:
            resultados.append({
                "id": sample["id"],
                "question": question,
                "faithfulness": 0.0,
                "relevance": 0.0,
                "error": str(e),
            })

    return resultados


def guardar_baseline(resultados: list[dict]):
    baseline = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": resultados,
        "avg_faithfulness": sum(r["faithfulness"] for r in resultados) / len(resultados),
        "avg_relevance": sum(r["relevance"] for r in resultados) / len(resultados),
    }
    BASELINE_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2))
    console.print(f"\n[green]Baseline guardado:[/] {BASELINE_PATH}")
    console.print(f"  avg_faithfulness: {baseline['avg_faithfulness']:.3f}")
    console.print(f"  avg_relevance:    {baseline['avg_relevance']:.3f}")
    return baseline


def comparar_con_baseline(resultados_actuales: list[dict], baseline: dict):
    """Compara los resultados actuales contra el baseline y reporta regresiones."""
    baseline_por_id = {r["id"]: r for r in baseline["samples"]}
    regresiones = []

    table = Table(title="Comparación con baseline", show_header=True, header_style="bold magenta")
    table.add_column("ID", width=6)
    table.add_column("Pregunta", max_width=30)
    table.add_column("Faith baseline")
    table.add_column("Faith actual")
    table.add_column("Δ Faith")
    table.add_column("Rel baseline")
    table.add_column("Rel actual")
    table.add_column("Δ Rel")
    table.add_column("Estado")

    for r in resultados_actuales:
        base = baseline_por_id.get(r["id"])
        if not base:
            continue

        delta_f = r["faithfulness"] - base["faithfulness"]
        delta_r = r["relevance"] - base["relevance"]
        es_regresion = delta_f < -REGRESSION_THRESHOLD or delta_r < -REGRESSION_THRESHOLD

        if es_regresion:
            regresiones.append(r["id"])
            estado = "[red]REGRESIÓN[/]"
        elif delta_f > REGRESSION_THRESHOLD or delta_r > REGRESSION_THRESHOLD:
            estado = "[green]MEJORA[/]"
        else:
            estado = "[dim]estable[/]"

        def fmt_delta(d):
            color = "green" if d > 0 else "red" if d < -0.01 else "dim"
            prefix = "+" if d > 0 else ""
            return f"[{color}]{prefix}{d:.2f}[/]"

        table.add_row(
            r["id"],
            r["question"][:28] + "...",
            f"{base['faithfulness']:.2f}",
            f"{r['faithfulness']:.2f}",
            fmt_delta(delta_f),
            f"{base['relevance']:.2f}",
            f"{r['relevance']:.2f}",
            fmt_delta(delta_r),
            estado,
        )

    console.print(table)

    # Resumen
    avg_f_actual = sum(r["faithfulness"] for r in resultados_actuales) / len(resultados_actuales)
    avg_r_actual = sum(r["relevance"] for r in resultados_actuales) / len(resultados_actuales)
    delta_f_global = avg_f_actual - baseline["avg_faithfulness"]
    delta_r_global = avg_r_actual - baseline["avg_relevance"]

    if regresiones:
        console.print(Panel(
            f"[red bold]⚠ REGRESIÓN DETECTADA en {len(regresiones)} sample(s): {', '.join(regresiones)}[/]\n\n"
            f"Δ Faithfulness global: {delta_f_global:+.3f}\n"
            f"Δ Relevance global:    {delta_r_global:+.3f}\n\n"
            f"[dim]Threshold: ±{REGRESSION_THRESHOLD}[/]",
            border_style="red",
        ))
    else:
        console.print(Panel(
            f"[green bold]✓ Sin regresiones detectadas[/]\n\n"
            f"Δ Faithfulness global: {delta_f_global:+.3f}\n"
            f"Δ Relevance global:    {delta_r_global:+.3f}",
            border_style="green",
        ))

    return regresiones


def main():
    console.rule("[bold blue]RAG Lab — Módulo 5.5: Regression Testing")

    if not DATASET_PATH.exists():
        console.print("[red]Ejecuta primero 24_eval_dataset.py[/]")
        return

    dataset = json.loads(DATASET_PATH.read_text())
    samples = dataset["samples"][:3]

    llm = get_llm()
    retriever = get_retriever()

    if not BASELINE_PATH.exists():
        console.print("\n[bold yellow]No existe baseline. Creando uno...[/]")
        console.print("[dim]Ejecutando evaluación para generar baseline...[/]")
        resultados = run_eval(samples, llm, retriever)
        guardar_baseline(resultados)
        console.print(Panel(
            "[bold]Baseline creado.[/]\n\n"
            "Para simular una regresión: modifica el prompt en rag/chain.py\n"
            "y vuelve a ejecutar este script. Detectará el cambio.\n\n"
            "[dim]En CI/CD: guarda el baseline en el repo y ejecuta este script\n"
            "como check antes de mergear un PR.[/]",
            border_style="blue",
        ))
    else:
        baseline = json.loads(BASELINE_PATH.read_text())
        console.print(f"\n[dim]Baseline cargado: {baseline['created_at']}[/]")
        console.print(f"[dim]avg_faithfulness={baseline['avg_faithfulness']:.3f}, "
                      f"avg_relevance={baseline['avg_relevance']:.3f}[/]")

        console.print("\n[dim]Ejecutando evaluación actual...[/]")
        resultados_actuales = run_eval(samples, llm, retriever)
        comparar_con_baseline(resultados_actuales, baseline)


if __name__ == "__main__":
    main()
