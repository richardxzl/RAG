"""
09_prompt_templates.py — Módulo 2.1: ChatPromptTemplate vs PromptTemplate

Demuestra las diferencias entre los dos tipos de templates de LangChain:
  - PromptTemplate: string plano, para modelos de completion (legacy)
  - ChatPromptTemplate: lista de mensajes, para chat models modernos

Incluye: partial variables, MessagesPlaceholder, inspección del template.
"""
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from rag.chain import get_llm

console = Console()


# ── Parte 1: PromptTemplate (string plano) ───────────────────────────────────

def demo_prompt_template():
    console.rule("[bold yellow]Parte 1 — PromptTemplate (completion style)")

    template = PromptTemplate.from_template(
        "Eres un asistente de cocina. Sugiere una receta para: {ingrediente}. "
        "Responde en máximo 3 oraciones."
    )

    console.print("\n[bold]Template creado:[/]")
    console.print(f"  input_variables: {template.input_variables}")

    # Formatear sin invocar el LLM — ver el string resultante
    prompt_formateado = template.format(ingrediente="tomates y queso")
    console.print(Panel(prompt_formateado, title="String formateado", border_style="dim"))

    # Invocar con el LLM
    llm = get_llm()
    chain = template | llm | StrOutputParser()
    respuesta = chain.invoke({"ingrediente": "tomates y queso"})
    console.print(Panel(respuesta, title="Respuesta del LLM", border_style="green"))

    console.print(
        "\n[dim]Nota: PromptTemplate envía un string sin rol 'system'. "
        "El LLM lo recibe como mensaje humano implícito.[/]"
    )


# ── Parte 2: ChatPromptTemplate (mensajes estructurados) ─────────────────────

def demo_chat_prompt_template():
    console.rule("[bold yellow]Parte 2 — ChatPromptTemplate (chat style)")

    # Forma 1: from_messages con tuplas (rol, contenido)
    template_chat = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente de cocina experto. Responde siempre en español, "
                   "en máximo 3 oraciones."),
        ("human", "Sugiere una receta para: {ingrediente}"),
    ])

    console.print("\n[bold]Template creado con from_messages:[/]")
    console.print(f"  input_variables: {template_chat.input_variables}")

    # Formatear — ver la lista de mensajes
    mensajes = template_chat.format_messages(ingrediente="tomates y queso")
    console.print("\n[bold]Mensajes formateados:[/]")
    for msg in mensajes:
        console.print(f"  [{msg.__class__.__name__}] {msg.content[:80]}")

    llm = get_llm()
    chain = template_chat | llm | StrOutputParser()
    respuesta = chain.invoke({"ingrediente": "tomates y queso"})
    console.print(Panel(respuesta, title="Respuesta (con rol system)", border_style="green"))

    # Forma 2: from_template — equivalente a PromptTemplate pero retorna ChatPromptTemplate
    template_simple = ChatPromptTemplate.from_template(
        "Eres un asistente de cocina. Sugiere una receta para: {ingrediente}"
    )
    console.print(
        "\n[dim]from_template() sobre ChatPromptTemplate crea un solo HumanMessage "
        "(sin system). Útil para prototipos rápidos.[/]"
    )


# ── Parte 3: Partial variables ────────────────────────────────────────────────

def demo_partial_variables():
    console.rule("[bold yellow]Parte 3 — Partial variables")

    template_base = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente que responde siempre en {idioma} y con tono {tono}."),
        ("human", "{pregunta}"),
    ])

    console.print(f"\nTemplate original — input_variables: {template_base.input_variables}")

    # Fijar idioma y tono, dejar pregunta libre
    template_es_formal = template_base.partial(idioma="español", tono="formal y profesional")
    console.print(f"Después de .partial() — input_variables: {template_es_formal.input_variables}")

    llm = get_llm()
    chain = template_es_formal | llm | StrOutputParser()
    respuesta = chain.invoke({"pregunta": "¿Cómo funciona la fotosíntesis?"})
    console.print(Panel(respuesta, title="Respuesta (idioma y tono pre-fijados)", border_style="green"))

    console.print(
        "\n[dim].partial() es útil para crear variantes de un template sin duplicarlo. "
        "El template base queda intacto.[/]"
    )


# ── Parte 4: MessagesPlaceholder (historial dinámico) ────────────────────────

def demo_messages_placeholder():
    console.rule("[bold yellow]Parte 4 — MessagesPlaceholder (historial)")

    template_chat = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente conversacional. Responde en español."),
        MessagesPlaceholder("chat_history"),  # ← aquí se insertan los mensajes previos
        ("human", "{pregunta}"),
    ])

    llm = get_llm()
    chain = template_chat | llm | StrOutputParser()

    # Sin historial
    respuesta_1 = chain.invoke({
        "chat_history": [],
        "pregunta": "Mi nombre es Carlos.",
    })
    console.print(Panel(respuesta_1, title="Turno 1 (sin historial)", border_style="blue"))

    # Con historial del turno anterior
    historial = [
        HumanMessage(content="Mi nombre es Carlos."),
        AIMessage(content=respuesta_1),
    ]
    respuesta_2 = chain.invoke({
        "chat_history": historial,
        "pregunta": "¿Recuerdas cómo me llamo?",
    })
    console.print(Panel(respuesta_2, title="Turno 2 (con historial)", border_style="green"))


# ── Tabla comparativa ─────────────────────────────────────────────────────────

def tabla_comparativa():
    console.rule("[bold yellow]Resumen comparativo")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Característica", style="cyan")
    table.add_column("PromptTemplate")
    table.add_column("ChatPromptTemplate")

    table.add_row("Tipo de output", "String", "Lista de BaseMessage")
    table.add_row("Roles (system/human/ai)", "No", "Sí")
    table.add_row("Ideal para", "Modelos completion (GPT-3, legacy)", "Chat models modernos")
    table.add_row("Historial conversacional", "Manual (concatenar string)", "MessagesPlaceholder")
    table.add_row("partial()", "Sí", "Sí")
    table.add_row("from_template()", "Sí", "Sí (crea HumanMessage)")
    table.add_row("from_messages()", "No", "Sí")

    console.print(table)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 2.1: Prompt Templates")
    demo_prompt_template()
    demo_chat_prompt_template()
    demo_partial_variables()
    demo_messages_placeholder()
    tabla_comparativa()


if __name__ == "__main__":
    main()
