"""
13_output_fixing.py — Módulo 2.5: Output fixing

Demuestra las 3 estrategias para manejar cuando el LLM no respeta el formato:
  1. OutputFixingParser: segunda llamada al LLM para corregir el output malo
  2. RetryWithErrorOutputParser: reintenta con el prompt original + el error
  3. Manejo manual con try/except: fallback explícito sin LLM extra

Caso de uso: parser de análisis de sentimiento con modelo Pydantic.
"""
import json
import logging
from typing import Literal
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import OutputFixingParser, RetryWithErrorOutputParser

from rag.chain import get_llm

console = Console()


# ── Modelo base ───────────────────────────────────────────────────────────────

class Analisis(BaseModel):
    sentimiento: Literal["positivo", "negativo", "neutro"]
    confianza: float = Field(ge=0.0, le=1.0, description="Nivel de confianza del 0.0 al 1.0")
    razon: str = Field(description="Razón breve del sentimiento detectado")


# ── Outputs malformados para simular fallos del LLM ──────────────────────────

OUTPUT_CAMPO_FALTANTE = json.dumps({
    "sentimiento": "positivo",
    "confianza": 0.9,
    # "razon" falta
})

OUTPUT_TIPO_INCORRECTO = json.dumps({
    "sentimiento": "positivo",
    "confianza": "muy alta",   # debería ser float
    "razon": "El tono es claramente entusiasta",
})

OUTPUT_VALOR_INVALIDO = json.dumps({
    "sentimiento": "excelente",   # no está en Literal
    "confianza": 0.85,
    "razon": "Reseña muy positiva",
})

OUTPUT_TEXTO_EXTRA = """
Aquí está mi análisis:

El texto tiene un tono claramente positivo.

```json
{"sentimiento": "positivo", "confianza": 0.88, "razon": "Tono entusiasta y positivo"}
```

Espero que esto sea útil.
"""


# ── Estrategia 1: OutputFixingParser ─────────────────────────────────────────

def demo_output_fixing():
    console.rule("[bold yellow]Estrategia 1 — OutputFixingParser")
    console.print(
        "[dim]Cuando el output es malo, hace una segunda llamada al LLM "
        "pasando el error para que lo corrija.[/]\n"
    )

    llm = get_llm()
    base_parser = PydanticOutputParser(pydantic_object=Analisis)

    # OutputFixingParser envuelve al parser base
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm,
    )

    casos = [
        ("Campo faltante", OUTPUT_CAMPO_FALTANTE),
        ("Tipo incorrecto (confianza)", OUTPUT_TIPO_INCORRECTO),
        ("Valor fuera de Literal", OUTPUT_VALOR_INVALIDO),
    ]

    for nombre, output_malo in casos:
        console.print(f"[bold]Caso: {nombre}[/]")
        console.print(f"  Input malo: [dim]{output_malo[:60]}...[/]")
        try:
            resultado = fixing_parser.parse(output_malo)
            console.print(
                f"  [green]Corregido:[/] sentimiento={resultado.sentimiento}, "
                f"confianza={resultado.confianza}, razon='{resultado.razon[:40]}'"
            )
        except Exception as e:
            console.print(f"  [red]OutputFixingParser también falló: {e}[/]")
        console.print()


# ── Estrategia 2: RetryWithErrorOutputParser ──────────────────────────────────

def demo_retry_with_error():
    console.rule("[bold yellow]Estrategia 2 — RetryWithErrorOutputParser")
    console.print(
        "[dim]Como OutputFixingParser pero también pasa el prompt original "
        "para que el LLM tenga más contexto al corregir.[/]\n"
    )

    llm = get_llm()
    base_parser = PydanticOutputParser(pydantic_object=Analisis)

    retry_parser = RetryWithErrorOutputParser.from_llm(
        parser=base_parser,
        llm=llm,
    )

    # Necesita el prompt original además del output malo
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analiza el sentimiento del texto.\n\n{format_instructions}"),
        ("human", "{texto}"),
    ])

    prompt_formateado = prompt.format_messages(
        formato=base_parser.get_format_instructions(),
        format_instructions=base_parser.get_format_instructions(),
        texto="El producto es fantástico, superó todas mis expectativas.",
    )

    console.print(f"[bold]Input malo:[/] [dim]{OUTPUT_TIPO_INCORRECTO[:60]}...[/]")
    try:
        resultado = retry_parser.parse_with_prompt(
            completion=OUTPUT_TIPO_INCORRECTO,
            prompt_value=prompt.format_prompt(
                format_instructions=base_parser.get_format_instructions(),
                texto="El producto es fantástico.",
            ),
        )
        console.print(
            f"[green]Corregido con contexto:[/] sentimiento={resultado.sentimiento}, "
            f"confianza={resultado.confianza}"
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


# ── Estrategia 3: Manejo manual ───────────────────────────────────────────────

def demo_manual():
    console.rule("[bold yellow]Estrategia 3 — Manejo manual con try/except")
    console.print(
        "[dim]Más predecible y sin costo de segunda llamada LLM. "
        "Retorna un objeto por defecto cuando el parser falla.[/]\n"
    )

    base_parser = PydanticOutputParser(pydantic_object=Analisis)

    def parse_safe(output: str) -> tuple[Analisis | None, bool]:
        """
        Intenta parsear. Si falla, retorna None y False.
        El llamador decide qué hacer con el fallo.
        """
        try:
            return base_parser.parse(output), True
        except (OutputParserException, Exception):
            return None, False

    def parse_with_default(output: str) -> Analisis:
        """
        Si el parser falla, retorna un objeto "neutro" predecible
        en vez de propagar la excepción.
        """
        resultado, ok = parse_safe(output)
        if ok:
            return resultado
        # Fallback: objeto por defecto
        return Analisis(
            sentimiento="neutro",
            confianza=0.0,
            razon="No se pudo analizar el output del LLM",
        )

    casos = [
        ("Output válido", json.dumps({"sentimiento": "positivo", "confianza": 0.9, "razon": "Tono positivo"})),
        ("Campo faltante", OUTPUT_CAMPO_FALTANTE),
        ("Tipo incorrecto", OUTPUT_TIPO_INCORRECTO),
    ]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Caso")
    table.add_column("Parseado?")
    table.add_column("Sentimiento")
    table.add_column("Confianza")

    for nombre, output in casos:
        r = parse_with_default(output)
        _, ok = parse_safe(output)
        table.add_row(
            nombre,
            "[green]Sí[/]" if ok else "[yellow]No → default[/]",
            r.sentimiento,
            str(r.confianza),
        )

    console.print(table)


# ── Tabla comparativa final ───────────────────────────────────────────────────

def tabla_comparativa():
    console.rule("[bold yellow]Comparativa de estrategias")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Estrategia", style="cyan")
    table.add_column("Llamadas LLM extra")
    table.add_column("Confiabilidad")
    table.add_column("Cuándo usar")

    table.add_row(
        "OutputFixingParser",
        "1 (si falla)",
        "Media",
        "Errores frecuentes de formato, modelo mediano",
    )
    table.add_row(
        "RetryWithErrorOutputParser",
        "1 (si falla)",
        "Media-Alta",
        "Errores semánticos, necesita contexto del prompt",
    )
    table.add_row(
        "Manual try/except",
        "0",
        "Alta (predecible)",
        "Cuando el costo de la llamada extra no vale",
    )
    table.add_row(
        ".with_structured_output()",
        "0",
        "Muy alta",
        "Mejor opción si el modelo soporta tool calling",
    )

    console.print(table)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 2.5: Output Fixing")
    demo_output_fixing()
    demo_retry_with_error()
    demo_manual()
    tabla_comparativa()


if __name__ == "__main__":
    main()
