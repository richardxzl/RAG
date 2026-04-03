"""
11_structured_output.py — Módulo 2.3: StructuredOutputParser

Demuestra cómo forzar respuestas JSON del LLM usando StructuredOutputParser:
  - Sin parser: el LLM responde libremente (texto extra, formato inconsistente)
  - Con parser: ResponseSchema define los campos, format_instructions van al prompt
  - En pipeline LCEL: prompt | llm | parser

Caso de uso: análisis estructurado de reseñas de productos.
"""
import json
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from rag.chain import get_llm

console = Console()

RESENAS = [
    "El auricular tiene un sonido increíble, la batería dura todo el día. "
    "Sin embargo, la diadema es un poco rígida y tras 2 horas empieza a molestar. "
    "Por el precio que tiene, la relación calidad-precio es muy buena.",

    "Decepcionante. La funda llegó con un arañazo visible y el cierre no encaja bien. "
    "El material parece barato comparado con las fotos. No lo recomiendo.",
]


# ── Parte 1: Sin parser (el problema) ────────────────────────────────────────

def demo_sin_parser(resena: str):
    console.rule("[bold yellow]Parte 1 — Sin parser (texto libre)")

    llm = get_llm()
    template = ChatPromptTemplate.from_messages([
        ("system", "Analiza la reseña y responde en JSON con: "
                   "sentimiento, puntuacion (1-5), aspectos_positivos, aspectos_negativos, resumen."),
        ("human", "{resena}"),
    ])
    chain = template | llm | StrOutputParser()
    respuesta = chain.invoke({"resena": resena})

    console.print(Panel(respuesta, title="Respuesta libre del LLM", border_style="dim"))

    # Intentar parsear — puede fallar o venir con texto extra
    try:
        # Buscar el JSON si viene envuelto en texto
        start = respuesta.find("{")
        end = respuesta.rfind("}") + 1
        if start >= 0:
            datos = json.loads(respuesta[start:end])
            console.print("[green]json.loads() funcionó — pero requirió buscar manualmente el JSON[/]")
        else:
            console.print("[red]No se encontró JSON en la respuesta[/]")
    except json.JSONDecodeError as e:
        console.print(f"[red]json.loads() falló: {e}[/]")


# ── Parte 2: Con StructuredOutputParser ──────────────────────────────────────

def build_parser():
    """
    ResponseSchema define cada campo: name es la key del dict resultante,
    description es lo que el LLM lee para saber qué poner ahí.
    La descripción importa — el LLM la usa como instrucción.
    """
    schemas = [
        ResponseSchema(name="sentimiento",
                       description="Sentimiento general: 'positivo', 'negativo' o 'neutro'"),
        ResponseSchema(name="puntuacion",
                       description="Puntuación numérica del 1 al 5 según la satisfacción"),
        ResponseSchema(name="aspectos_positivos",
                       description="Lista de aspectos positivos mencionados. Array de strings."),
        ResponseSchema(name="aspectos_negativos",
                       description="Lista de aspectos negativos mencionados. Array de strings."),
        ResponseSchema(name="resumen",
                       description="Resumen de la reseña en máximo una oración"),
    ]
    return StructuredOutputParser.from_response_schemas(schemas)


def demo_con_parser(resena: str):
    console.rule("[bold yellow]Parte 2 — Con StructuredOutputParser")

    llm = get_llm()
    parser = build_parser()

    # Las format_instructions le dicen al LLM exactamente cómo formatear el JSON
    format_instructions = parser.get_format_instructions()
    console.print("\n[bold]format_instructions (lo que ve el LLM):[/]")
    console.print(Panel(format_instructions[:400] + "...", border_style="dim"))

    template = ChatPromptTemplate.from_messages([
        ("system", "Analiza la reseña del producto.\n\n{format_instructions}"),
        ("human", "{resena}"),
    ])

    # prompt | llm | parser — el parser convierte el string en dict
    chain = template | llm | parser

    resultado = chain.invoke({
        "resena": resena,
        "format_instructions": format_instructions,
    })

    console.print("\n[bold]Dict resultante:[/]")
    console.print(JSON(json.dumps(resultado, ensure_ascii=False, indent=2)))

    # Acceso tipado — aunque los valores son strings, el dict es predecible
    console.print(f"\n  sentimiento: [cyan]{resultado['sentimiento']}[/]")
    console.print(f"  puntuacion:  [cyan]{resultado['puntuacion']}[/]")
    console.print(f"  positivos:   [cyan]{resultado['aspectos_positivos']}[/]")


# ── Parte 3: Pipeline con múltiples reseñas ──────────────────────────────────

def demo_pipeline_completo():
    console.rule("[bold yellow]Parte 3 — Pipeline con múltiples reseñas")

    llm = get_llm()
    parser = build_parser()
    format_instructions = parser.get_format_instructions()

    template = ChatPromptTemplate.from_messages([
        ("system", "Analiza la reseña del producto.\n\n{format_instructions}"),
        ("human", "{resena}"),
    ])

    chain = template | llm | parser

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Reseña", max_width=35, style="dim")
    table.add_column("Sentimiento")
    table.add_column("Puntuación")
    table.add_column("Positivos")
    table.add_column("Negativos")

    for resena in RESENAS:
        resultado = chain.invoke({
            "resena": resena,
            "format_instructions": format_instructions,
        })
        positivos = ", ".join(resultado.get("aspectos_positivos", []))[:40]
        negativos = ", ".join(resultado.get("aspectos_negativos", []))[:40]
        table.add_row(
            resena[:32] + "...",
            resultado.get("sentimiento", "?"),
            str(resultado.get("puntuacion", "?")),
            positivos,
            negativos,
        )

    console.print(table)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 2.3: StructuredOutputParser")
    demo_sin_parser(RESENAS[0])
    demo_con_parser(RESENAS[0])
    demo_pipeline_completo()


if __name__ == "__main__":
    main()
