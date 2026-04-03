"""
33_error_handling.py — Módulo 6.5: Error handling y graceful degradation

Demuestra cómo el pipeline RAG se degrada con gracia cuando fallan componentes:
  - Redis down → el sistema funciona sin cache (más lento, pero funciona)
  - ChromaDB no disponible → respuesta directa del LLM sin RAG
  - API key inválida → error claro con mensaje de usuario apropiado
  - Timeout del LLM → respuesta de fallback predefinida

Principio: el usuario nunca debe ver un stack trace. Siempre hay una respuesta.
"""
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag.chain import get_llm, QUERY_PROMPT, format_docs

console = Console()

# Respuestas de fallback para el usuario
FALLBACK_SIN_CONTEXTO = (
    "Lo siento, no pude acceder a la base de conocimiento en este momento. "
    "Puedo intentar responder con mi conocimiento general: "
)
FALLBACK_ERROR_TOTAL = (
    "Lo siento, estoy experimentando problemas técnicos. "
    "Por favor, inténtalo de nuevo en unos minutos."
)


# ── Degradación 1: Redis down ─────────────────────────────────────────────────

def get_cache_seguro():
    """
    Intenta conectar al cache. Si falla, retorna None.
    El pipeline debe funcionar sin cache (más lento, no roto).
    """
    try:
        from rag.cache import SemanticCache
        cache = SemanticCache()
        # Test de conexión
        cache.stats()
        return cache
    except Exception as e:
        console.print(f"  [yellow]Cache no disponible: {e}[/]")
        console.print("  [dim]Continuando sin cache (degradado)[/]")
        return None


def query_con_cache_opcional(question: str) -> tuple[str, str]:
    """
    Pipeline que funciona con o sin cache.
    Retorna (respuesta, modo) donde modo es "cache", "rag" o "degradado".
    """
    cache = get_cache_seguro()

    # Intentar cache primero
    if cache:
        try:
            cached = cache.get(question)
            if cached:
                return cached, "cache"
        except Exception:
            pass  # cache falló a mitad — continuar sin él

    # RAG normal
    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.invoke(question)
        context = format_docs(docs)

        llm = get_llm()
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | QUERY_PROMPT
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke({"question": question, "docs": docs})

        if cache:
            try:
                cache.set(question, answer)
            except Exception:
                pass  # no pasa nada si el cache falla al escribir

        return answer, "rag"
    except Exception as e:
        return f"{FALLBACK_ERROR_TOTAL} (detalle técnico: {type(e).__name__})", "error"


# ── Degradación 2: ChromaDB no disponible ─────────────────────────────────────

def query_con_rag_opcional(question: str) -> tuple[str, str]:
    """
    Intenta RAG. Si el retriever falla, responde solo con el LLM.
    La respuesta es de menor calidad pero el sistema no cae.
    """
    llm = get_llm()

    # Intento 1: RAG completo
    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.invoke(question)

        if not docs:
            raise ValueError("No se recuperaron documentos")

        context = format_docs(docs)
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | QUERY_PROMPT
            | llm
            | StrOutputParser()
        )
        return chain.invoke({"question": question, "docs": docs}), "rag_completo"

    except Exception as e:
        console.print(f"  [yellow]RAG falló ({type(e).__name__}): {e}[/]")
        console.print("  [dim]Degradando a LLM sin contexto[/]")

    # Intento 2: LLM directo sin RAG
    try:
        prompt_sin_rag = ChatPromptTemplate.from_messages([
            ("system",
             "Eres un asistente útil. Responde basándote en tu conocimiento general. "
             "Si no sabes la respuesta, dilo claramente."),
            ("human", "{question}"),
        ])
        chain = prompt_sin_rag | llm | StrOutputParser()
        answer = chain.invoke({"question": question})
        return f"{FALLBACK_SIN_CONTEXTO}{answer}", "llm_sin_rag"

    except Exception as e:
        console.print(f"  [red]LLM también falló: {e}[/]")
        return FALLBACK_ERROR_TOTAL, "error_total"


# ── Degradación 3: API key inválida ───────────────────────────────────────────

def query_con_manejo_auth(question: str) -> tuple[str, str]:
    """Maneja errores de autenticación con mensaje claro para el usuario."""
    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.invoke(question)
        context = format_docs(docs)
        llm = get_llm()
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | QUERY_PROMPT
            | llm
            | StrOutputParser()
        )
        return chain.invoke({"question": question, "docs": docs}), "ok"

    except Exception as e:
        error_str = str(e).lower()

        # Mapear errores técnicos a mensajes de usuario
        if "api_key" in error_str or "authentication" in error_str or "401" in error_str:
            return "Error de configuración del servicio. Contacta al administrador.", "auth_error"
        elif "rate_limit" in error_str or "429" in error_str:
            return "Servicio temporalmente saturado. Por favor, espera unos segundos e inténtalo de nuevo.", "rate_limit"
        elif "timeout" in error_str or "timed out" in error_str:
            return "La respuesta tardó demasiado. Inténtalo con una pregunta más específica.", "timeout"
        else:
            return FALLBACK_ERROR_TOTAL, "unknown_error"


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]RAG Lab — Módulo 6.5: Error Handling")

    console.print(Panel(
        "[bold]Principio de graceful degradation:[/]\n\n"
        "El sistema tiene niveles de capacidad:\n"
        "  Nivel 3 (óptimo):  cache + RAG + LLM\n"
        "  Nivel 2 (degradado): RAG + LLM (sin cache)\n"
        "  Nivel 1 (mínimo):  solo LLM (sin contexto)\n"
        "  Nivel 0 (error):   mensaje de error claro\n\n"
        "[dim]El usuario nunca ve un stack trace.\n"
        "Cada nivel de degradación es transparente para el operador (logs)[/]",
        border_style="blue",
    ))

    question = "¿De qué trata el documento?"

    # Demo 1: Cache opcional
    console.rule("[bold yellow]Demo 1: Cache opcional (Redis puede estar down)")
    answer, modo = query_con_cache_opcional(question)
    console.print(Panel(
        f"[dim]Modo: {modo}[/]\n{answer[:200]}",
        title="Respuesta",
        border_style="green",
    ))

    # Demo 2: RAG opcional
    console.rule("[bold yellow]Demo 2: RAG opcional (ChromaDB puede estar down)")
    answer, modo = query_con_rag_opcional(question)
    console.print(Panel(
        f"[dim]Modo: {modo}[/]\n{answer[:200]}",
        title="Respuesta",
        border_style="green",
    ))

    # Demo 3: Manejo de auth
    console.rule("[bold yellow]Demo 3: Manejo de errores de autenticación y rate limits")
    answer, modo = query_con_manejo_auth(question)
    console.print(Panel(
        f"[dim]Modo: {modo}[/]\n{answer[:200]}",
        title="Respuesta",
        border_style="green",
    ))

    # Tabla de estrategias
    table = Table(title="Estrategias de error handling", show_header=True, header_style="bold magenta")
    table.add_column("Fallo", style="red")
    table.add_column("Estrategia")
    table.add_column("Experiencia del usuario")

    table.add_row("Redis down", "Continuar sin cache", "Mismo resultado, más lento")
    table.add_row("ChromaDB down", "Degradar a LLM directo", "Respuesta sin contexto del documento")
    table.add_row("API key inválida", "Mensaje de error claro", "Contactar administrador")
    table.add_row("Rate limit", "Mensaje de retry", "Esperar y reintentar")
    table.add_row("Timeout LLM", "Mensaje de timeout", "Pregunta más específica")
    table.add_row("Error desconocido", "Fallback genérico", "Reintentar en unos minutos")
    console.print(table)


if __name__ == "__main__":
    main()
