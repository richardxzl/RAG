"""
24_eval_dataset.py — Módulo 5.1: Dataset de evaluación

Crea y gestiona un dataset de preguntas + respuestas esperadas (ground truth).
El dataset es la base para todas las métricas de los módulos 5.2 a 5.5.

Formato del dataset: JSON con lista de EvalSample
  - question: pregunta de evaluación
  - ground_truth: respuesta de referencia (escrita por un humano o experto)
  - context_docs: documentos relevantes esperados (opcional)
  - metadata: etiquetas para categorizar (tipo, dificultad)
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

DATASET_PATH = Path("eval_dataset.json")

# Dataset de evaluación (ground truth manual)
# En producción: generado por expertos del dominio
EVAL_SAMPLES = [
    {
        "id": "q001",
        "question": "¿Qué es LCEL?",
        "ground_truth": "LCEL (LangChain Expression Language) es el sistema de composición "
                        "moderno de LangChain que usa el operador pipe (|) para conectar "
                        "componentes Runnable de forma declarativa.",
        "metadata": {"tipo": "conceptual", "dificultad": "baja", "modulo": "1"},
    },
    {
        "id": "q002",
        "question": "¿Qué ventajas tiene LCEL sobre las chains legacy?",
        "ground_truth": "LCEL ofrece streaming nativo, batch nativo, composabilidad total "
                        "con el operador |, y una interfaz unificada para todos los componentes. "
                        "Las chains legacy tenían APIs inconsistentes y no soportaban streaming fácilmente.",
        "metadata": {"tipo": "comparativa", "dificultad": "media", "modulo": "1"},
    },
    {
        "id": "q003",
        "question": "¿Cuándo usar RunnableBranch?",
        "ground_truth": "RunnableBranch se usa cuando el pipeline necesita tomar decisiones "
                        "condicionales: ejecutar un chain diferente según el tipo de input. "
                        "Por ejemplo, routing por intención del usuario.",
        "metadata": {"tipo": "aplicación", "dificultad": "media", "modulo": "1"},
    },
    {
        "id": "q004",
        "question": "¿Qué diferencia hay entre PromptTemplate y ChatPromptTemplate?",
        "ground_truth": "PromptTemplate genera un string plano para modelos de completion. "
                        "ChatPromptTemplate genera una lista de mensajes con roles (system, human, ai) "
                        "para chat models modernos como Claude o GPT-4.",
        "metadata": {"tipo": "comparativa", "dificultad": "baja", "modulo": "2"},
    },
    {
        "id": "q005",
        "question": "¿Qué es MMR en el contexto de retrievers?",
        "ground_truth": "MMR (Maximum Marginal Relevance) es una estrategia de búsqueda que "
                        "balancea relevancia con diversidad. Evita retornar chunks redundantes "
                        "al penalizar candidatos muy similares a los ya seleccionados.",
        "metadata": {"tipo": "conceptual", "dificultad": "media", "modulo": "3"},
    },
    {
        "id": "q006",
        "question": "¿Cómo funciona el Parent-Child Retriever?",
        "ground_truth": "Usa chunks pequeños para buscar con precisión (mayor similitud coseno) "
                        "pero retorna el chunk padre grande al LLM para darle más contexto. "
                        "Los hijos se vectorizan en ChromaDB, los padres se guardan en un docstore.",
        "metadata": {"tipo": "técnico", "dificultad": "alta", "modulo": "3"},
    },
    {
        "id": "q007",
        "question": "¿Por qué usar SemanticChunker en lugar de RecursiveCharacterTextSplitter?",
        "ground_truth": "SemanticChunker corta por cambios semánticos reales (cuando el tema cambia) "
                        "en vez de por tamaño fijo. Genera chunks temáticamente coherentes, "
                        "aunque es más lento porque vectoriza las oraciones durante la ingesta.",
        "metadata": {"tipo": "comparativa", "dificultad": "media", "modulo": "4"},
    },
    {
        "id": "q008",
        "question": "¿Qué es el Contextual Compression Retriever?",
        "ground_truth": "Es un wrapper que aplica un compressor a los chunks después de recuperarlos. "
                        "LLMChainExtractor extrae solo la parte relevante de cada chunk. "
                        "EmbeddingsFilter elimina chunks con similitud baja al query.",
        "metadata": {"tipo": "técnico", "dificultad": "media", "modulo": "3"},
    },
]


def guardar_dataset(samples: list[dict], path: Path):
    dataset = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(samples),
        "samples": samples,
    }
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"  [green]Dataset guardado:[/] {path} ({len(samples)} samples)")


def cargar_dataset(path: Path) -> dict:
    if not path.exists():
        console.print(f"[red]Dataset no encontrado: {path}[/]")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def mostrar_dataset(dataset: dict):
    samples = dataset.get("samples", [])
    table = Table(
        title=f"Dataset de evaluación — v{dataset.get('version', '?')} ({len(samples)} samples)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", width=6)
    table.add_column("Tipo", width=12)
    table.add_column("Dificultad", width=10)
    table.add_column("Módulo", width=7)
    table.add_column("Pregunta", max_width=45)
    table.add_column("Ground truth (preview)", max_width=45)

    for s in samples:
        meta = s.get("metadata", {})
        table.add_row(
            s["id"],
            meta.get("tipo", "?"),
            meta.get("dificultad", "?"),
            meta.get("modulo", "?"),
            s["question"],
            s["ground_truth"][:60] + "...",
        )
    console.print(table)


def estadisticas_dataset(dataset: dict):
    samples = dataset.get("samples", [])
    tipos = {}
    dificultades = {}
    modulos = {}
    for s in samples:
        meta = s.get("metadata", {})
        tipos[meta.get("tipo", "?")] = tipos.get(meta.get("tipo", "?"), 0) + 1
        dificultades[meta.get("dificultad", "?")] = dificultades.get(meta.get("dificultad", "?"), 0) + 1
        modulos[meta.get("modulo", "?")] = modulos.get(meta.get("modulo", "?"), 0) + 1

    console.print("\n[bold]Distribución del dataset:[/]")
    for label, counts in [("Por tipo", tipos), ("Por dificultad", dificultades), ("Por módulo", modulos)]:
        t = Table(title=label, show_header=True, header_style="bold cyan")
        t.add_column("Valor")
        t.add_column("Count")
        for k, v in sorted(counts.items()):
            t.add_row(k, str(v))
        console.print(t)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 5.1: Dataset de Evaluación")

    console.print(Panel(
        "[bold]¿Por qué necesitas un dataset de evaluación?[/]\n\n"
        "Sin ground truth no puedes medir si el RAG funciona bien.\n"
        "El dataset es la base de verdad contra la que comparas las respuestas.\n\n"
        "Un buen dataset tiene:\n"
        "  · Preguntas representativas del uso real\n"
        "  · Respuestas escritas por expertos del dominio\n"
        "  · Variedad de tipos (conceptual, técnico, comparativo)\n"
        "  · Variedad de dificultad\n\n"
        "[dim]Regla: si no puedes definir qué es una respuesta correcta,\n"
        "tampoco puedes medir si el sistema la produce.[/]",
        border_style="blue",
    ))

    # Guardar y cargar
    guardar_dataset(EVAL_SAMPLES, DATASET_PATH)
    dataset = cargar_dataset(DATASET_PATH)

    mostrar_dataset(dataset)
    estadisticas_dataset(dataset)

    console.print(Panel(
        f"[green]Dataset guardado en:[/] {DATASET_PATH}\n\n"
        "Los módulos 5.2, 5.3 y 5.4 usan este archivo como entrada.\n"
        "Para extenderlo: agrega más samples a EVAL_SAMPLES y vuelve a ejecutar.",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
