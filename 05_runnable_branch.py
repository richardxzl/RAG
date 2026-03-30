"""
05_runnable_branch.py — RunnableBranch: condicionales en el pipeline LCEL

Demuestra cómo bifurcar un pipeline LCEL según la intención detectada
en la pregunta del usuario. El clasificador es puramente sintáctico
(sin llamadas al LLM) para mantener la latencia mínima.

Tres ramas posibles:
  1. fuera_de_scope  → el LLM responde que no puede ayudar (sin RAG)
  2. conversacional  → el LLM responde directamente (sin RAG)
  3. factual         → pipeline RAG completo (rama por defecto)
"""
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.chain import get_llm, QUERY_PROMPT, format_docs
from rag.retriever import get_retriever

console = Console()


# ── Prompts para ramas sin RAG ───────────────────────────────────────────────

FUERA_DE_SCOPE_PROMPT = ChatPromptTemplate.from_template(
    """Eres un asistente especializado en documentación técnica.
El usuario te ha hecho una pregunta que está fuera de tu área de conocimiento.

Pregunta: {question}

Responde amablemente explicando que solo puedes ayudar con preguntas
sobre la documentación técnica disponible. Sugiere reformular la pregunta
si cree que sí es relevante. Responde en español, de forma breve."""
)

CONVERSACIONAL_PROMPT = ChatPromptTemplate.from_template(
    """Eres un asistente amigable y útil que responde consultas generales.

Pregunta o saludo: {question}

Responde de forma natural y cálida, en español. Si es un saludo,
saluda de vuelta y ofrece ayuda con la documentación técnica disponible."""
)


# ── Clasificador de intención (sin LLM) ─────────────────────────────────────

# Palabras clave que indican que la pregunta está fuera del dominio técnico.
# En producción esto podría ser un modelo de clasificación liviano, pero para
# mantener latencia mínima y determinismo en el pipeline, usamos reglas simples.
FUERA_DE_SCOPE_KEYWORDS = {
    "clima", "tiempo", "temperatura", "pronóstico",
    "receta", "cocina", "ingredientes",
    "deporte", "fútbol", "baloncesto", "tenis",
    "película", "serie", "netflix",
    "política", "elecciones", "presidente",
    "precio", "bolsa", "acciones", "criptomoneda",
    "chiste", "broma", "historia",
}

# Palabras clave o patrones que indican conversación casual / saludos.
CONVERSACIONAL_KEYWORDS = {
    "hola", "buenos días", "buenas tardes", "buenas noches",
    "gracias", "adios", "hasta luego", "chao", "bye",
    "cómo estás", "como estas", "qué tal", "que tal",
    "quién eres", "quien eres", "qué eres", "que eres",
    "ayuda", "help",
}


def detectar_intencion(input_data: dict) -> str:
    """
    Clasifica la pregunta en una de tres intenciones:
      - "fuera_de_scope": el tema no es de documentación técnica
      - "conversacional": saludo o conversación general
      - "factual": pregunta técnica que requiere RAG

    ¿Por qué no usar el LLM para clasificar?
    Llamar al LLM para clasificar agrega ~500ms de latencia adicional
    ANTES de la respuesta real. Para intenciones simples, las reglas
    keyword-based son más rápidas, deterministas y gratuitas.
    Para sistemas más complejos, considera un modelo de clasificación
    local pequeño (fastText, SetFit) antes de escalar al LLM.
    """
    question = input_data.get("question", "").lower().strip()
    palabras = set(question.split())

    if palabras & FUERA_DE_SCOPE_KEYWORDS:
        return "fuera_de_scope"

    if palabras & CONVERSACIONAL_KEYWORDS:
        return "conversacional"

    # Si la pregunta es muy corta (menos de 4 palabras) y no parece técnica,
    # la tratamos como conversacional en lugar de mandarla al RAG sin contexto.
    if len(palabras) < 4:
        return "conversacional"

    return "factual"


# ── Ramas del pipeline ────────────────────────────────────────────────────────

def build_rama_fuera_de_scope(llm):
    """
    Rama para preguntas fuera del dominio.
    El LLM responde con un mensaje de cortesía explicando la limitación.
    No llama al retriever — no tiene sentido buscar docs sobre el clima.
    """
    return FUERA_DE_SCOPE_PROMPT | llm | StrOutputParser()


def build_rama_conversacional(llm):
    """
    Rama para saludos y conversación general.
    El LLM responde directamente sin contexto de documentos.
    No tiene sentido recuperar docs para "hola, ¿cómo estás?".
    """
    return CONVERSACIONAL_PROMPT | llm | StrOutputParser()


def build_rama_factual(llm, retriever):
    """
    Rama para preguntas técnicas — pipeline RAG completo.
    Esta es la rama por defecto: cualquier pregunta que no sea
    conversacional ni fuera de scope pasa por aquí.
    """
    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
        )
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )


# ── Pipeline con RunnableBranch ───────────────────────────────────────────────

def build_router_pipeline():
    """
    Ensambla el pipeline completo con RunnableBranch.

    RunnableBranch evalúa condiciones en orden. La primera condición
    que retorne True ejecuta su runnable asociado. Si ninguna coincide,
    ejecuta el runnable por defecto (rama factual).

    Estructura:
        RunnableBranch(
            (condicion_1, runnable_1),
            (condicion_2, runnable_2),
            runnable_default,  ← sin condición, es el fallback
        )
    """
    llm = get_llm()
    retriever = get_retriever()

    rama_fuera = build_rama_fuera_de_scope(llm)
    rama_conv  = build_rama_conversacional(llm)
    rama_fact  = build_rama_factual(llm, retriever)

    branch = RunnableBranch(
        # Condición 1: pregunta fuera del dominio técnico
        (
            lambda x: detectar_intencion(x) == "fuera_de_scope",
            rama_fuera,
        ),
        # Condición 2: saludo o conversación casual
        (
            lambda x: detectar_intencion(x) == "conversacional",
            rama_conv,
        ),
        # Default: pregunta factual → RAG completo
        rama_fact,
    )

    return branch


# ── Helpers de presentación ───────────────────────────────────────────────────

RAMA_ESTILOS = {
    "fuera_de_scope": ("rojo", "red", "[red]Fuera de scope[/red]"),
    "conversacional": ("azul", "blue", "[blue]Conversacional[/blue]"),
    "factual":        ("verde", "green", "[green]Factual (RAG)[/green]"),
}


def mostrar_resultado(pregunta: str, respuesta: str, intencion: str):
    """Imprime la pregunta, la rama detectada y la respuesta con rich."""
    _, border, etiqueta = RAMA_ESTILOS[intencion]

    console.print(f"\n[bold]Pregunta:[/] {pregunta}")
    console.print(f"[bold]Rama detectada:[/] {etiqueta}")
    console.print(Panel(
        Markdown(respuesta),
        title=f"Respuesta — {etiqueta}",
        border_style=border,
    ))


# ── Demo ─────────────────────────────────────────────────────────────────────

def run_demo():
    console.rule("[bold blue]RAG Lab — RunnableBranch (1.2)")

    console.print(
        "\n[dim]Clasificación de intención basada en reglas keyword.\n"
        "Cada pregunta toma una rama diferente del pipeline.[/]\n"
    )

    pipeline = build_router_pipeline()

    # Casos de prueba: uno por cada rama
    casos = [
        {
            "question": "¿Cuál es el pronóstico del tiempo para mañana en Madrid?",
            "intencion_esperada": "fuera_de_scope",
        },
        {
            "question": "Hola, ¿cómo estás? ¿En qué me puedes ayudar?",
            "intencion_esperada": "conversacional",
        },
        {
            "question": "¿De qué trata el documento principal?",
            "intencion_esperada": "factual",
        },
        {
            "question": "¿Cuál es la receta del gazpacho andaluz?",
            "intencion_esperada": "fuera_de_scope",
        },
        {
            "question": "Gracias por tu ayuda, fue muy útil.",
            "intencion_esperada": "conversacional",
        },
    ]

    for caso in casos:
        pregunta = caso["question"]
        intencion = detectar_intencion({"question": pregunta})

        # Verificar que la detección coincide con lo esperado
        esperada = caso["intencion_esperada"]
        estado = "[green]OK[/green]" if intencion == esperada else f"[red]MISMATCH (esperaba {esperada})[/red]"
        console.print(f"[dim]Clasificador: {intencion} {estado}[/dim]")

        respuesta = pipeline.invoke({"question": pregunta})
        mostrar_resultado(pregunta, respuesta, intencion)

    # ── Tabla resumen ──
    console.print("\n[bold]Resumen: ¿Cuándo usar RunnableBranch?[/]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rama", style="cyan")
    table.add_column("Condición")
    table.add_column("Pipeline ejecutado")
    table.add_column("¿Llama al retriever?")

    table.add_row(
        "fuera_de_scope",
        "keywords de dominio externo",
        "FUERA_DE_SCOPE_PROMPT → LLM",
        "[red]No[/red]",
    )
    table.add_row(
        "conversacional",
        "keywords de saludo / < 4 palabras",
        "CONVERSACIONAL_PROMPT → LLM",
        "[red]No[/red]",
    )
    table.add_row(
        "factual (default)",
        "ninguna condición previa",
        "retriever → QUERY_PROMPT → LLM",
        "[green]Sí[/green]",
    )

    console.print(table)

    console.print(
        "\n[dim]Tip: RunnableBranch evalúa las condiciones en orden.\n"
        "La primera que retorna True gana. El default no tiene condición.[/]"
    )


if __name__ == "__main__":
    run_demo()
