"""
12_pydantic_output.py — Módulo 2.4: PydanticOutputParser

Demuestra validación tipada de output del LLM usando Pydantic:
  - Modelo Pydantic con tipos, Literal, Optional y validadores
  - PydanticOutputParser genera el JSON schema que el LLM entiende
  - Validación automática al parsear — lanza ValidationError si los tipos son incorrectos
  - Pipeline LCEL completo: prompt | llm | parser

Caso de uso: extraer información estructurada de textos de noticias.
"""
import json
import logging
from typing import Optional, Literal
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.table import Table

from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from rag.chain import get_llm

console = Console()


# ── Modelo Pydantic ───────────────────────────────────────────────────────────

class NoticiaEstructurada(BaseModel):
    """
    El modelo define qué queremos extraer. Los Field(description=...) son
    leídos por el LLM — son instrucciones, no solo documentación.
    """
    titulo: str = Field(description="Título conciso de la noticia en máximo 10 palabras")
    categoria: Literal["tecnología", "economía", "política", "deportes", "cultura", "otro"] = Field(
        description="Categoría temática principal de la noticia"
    )
    entidades: list[str] = Field(
        description="Personas, empresas u organizaciones mencionadas en el texto"
    )
    fecha_mencionada: Optional[str] = Field(
        default=None,
        description="Fecha mencionada en el texto (formato libre), null si no hay ninguna"
    )
    resumen: str = Field(description="Resumen en exactamente 2 oraciones")
    relevancia: int = Field(
        ge=1, le=10,
        description="Relevancia del 1 (baja) al 10 (alta) según el impacto de la noticia"
    )

    @field_validator("categoria", mode="before")
    @classmethod
    def normalizar_categoria(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v


NOTICIAS = [
    """
    La empresa tecnológica Anthropic anunció ayer el lanzamiento de Claude 4,
    su modelo de inteligencia artificial más avanzado hasta la fecha. Según el CEO
    Dario Amodei, el modelo supera a sus competidores en tareas de razonamiento complejo.
    La noticia generó una subida del 3% en las acciones del sector tech en Wall Street.
    """,

    """
    El Real Madrid venció al Barcelona por 2-1 en el clásico disputado el pasado sábado.
    Los goles fueron obra de Vinicius Jr. en el minuto 23 y Bellingham en el 78.
    El equipo madridista se sitúa ahora 5 puntos por encima de su rival en la clasificación.
    """,
]


# ── Parte 1: Setup y format_instructions ─────────────────────────────────────

def demo_format_instructions():
    console.rule("[bold yellow]Parte 1 — Format instructions (lo que el LLM ve)")

    parser = PydanticOutputParser(pydantic_object=NoticiaEstructurada)
    instrucciones = parser.get_format_instructions()

    console.print(Panel(
        instrucciones[:600] + "\n[dim]...[/]",
        title="JSON Schema generado automáticamente",
        border_style="dim"
    ))
    console.print(
        "\n[dim]El schema incluye tipos, descripciones y restricciones "
        "(ge=1, le=10, Literal). El LLM lo lee como especificación.[/]"
    )


# ── Parte 2: Pipeline completo ────────────────────────────────────────────────

def demo_pipeline(noticia: str) -> NoticiaEstructurada:
    """
    prompt | llm | parser
    El parser convierte el string JSON en un objeto NoticiaEstructurada validado.
    """
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=NoticiaEstructurada)

    template = ChatPromptTemplate.from_messages([
        ("system",
         "Extrae información estructurada del texto de la noticia.\n\n"
         "{format_instructions}"),
        ("human", "{noticia}"),
    ])

    chain = template | llm | parser

    return chain.invoke({
        "noticia": noticia,
        "format_instructions": parser.get_format_instructions(),
    })


# ── Parte 3: Validación en acción ────────────────────────────────────────────

def demo_validacion():
    console.rule("[bold yellow]Parte 3 — Validación de tipos")

    parser = PydanticOutputParser(pydantic_object=NoticiaEstructurada)

    # JSON válido
    json_valido = json.dumps({
        "titulo": "Anthropic lanza Claude 4",
        "categoria": "tecnología",
        "entidades": ["Anthropic", "Dario Amodei"],
        "fecha_mencionada": None,
        "resumen": "Anthropic lanzó su modelo más avanzado. Superó a competidores en razonamiento.",
        "relevancia": 8,
    })

    # JSON con tipo incorrecto: relevancia como string en vez de int
    json_invalido = json.dumps({
        "titulo": "Noticia de prueba",
        "categoria": "otro",
        "entidades": [],
        "fecha_mencionada": None,
        "resumen": "Resumen. Segunda oración.",
        "relevancia": "muy alta",  # ← debería ser int 1-10
    })

    console.print("\n[bold]Parseando JSON válido:[/]")
    try:
        resultado = parser.parse(json_valido)
        console.print(f"  [green]OK[/] — relevancia es int: {type(resultado.relevancia).__name__} = {resultado.relevancia}")
    except Exception as e:
        console.print(f"  [red]Error: {e}[/]")

    console.print("\n[bold]Parseando JSON con tipo incorrecto:[/]")
    try:
        parser.parse(json_invalido)
    except (ValidationError, Exception) as e:
        console.print(f"  [red]ValidationError capturado:[/]")
        console.print(Panel(str(e)[:300], border_style="red"))
        console.print("  [dim]Pydantic rechaza el valor antes de que llegue a tu código.[/]")


# ── Demo principal ────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 2.4: PydanticOutputParser")

    demo_format_instructions()

    console.rule("[bold yellow]Parte 2 — Pipeline completo")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Campo", style="cyan")
    table.add_column("Noticia 1")
    table.add_column("Noticia 2")

    resultados = []
    for noticia in NOTICIAS:
        console.print(f"\n[dim]Procesando noticia...[/]")
        r = demo_pipeline(noticia)
        resultados.append(r)

        console.print(Panel(
            JSON(r.model_dump_json(indent=2)),
            title=f"[bold]{r.titulo}[/]",
            border_style="green",
        ))

    if len(resultados) == 2:
        r1, r2 = resultados
        campos = ["titulo", "categoria", "relevancia", "fecha_mencionada"]
        for campo in campos:
            table.add_row(
                campo,
                str(getattr(r1, campo)),
                str(getattr(r2, campo)),
            )
        console.print(table)

    demo_validacion()


if __name__ == "__main__":
    main()
