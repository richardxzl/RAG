"""
03_chat.py — Chat RAG interactivo con LCEL + Memoria + Cache
"""
import logging
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from rag.config import EMBEDDING_MODEL, CHROMA_DIR, MEMORY_WINDOW
from rag.chain import build_chat_chain

console = Console()


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    console.rule("[bold blue]🤖 RAG Chat Interactivo (LCEL + Memoria + Cache)")
    console.print("Escribe tus preguntas. Comandos especiales:")
    console.print("  [cyan]/debug[/]   — Activa/desactiva ver chunks recuperados")
    console.print("  [cyan]/stats[/]   — Muestra estadísticas")
    console.print("  [cyan]/clear[/]   — Limpia historial de conversación")
    console.print("  [cyan]/flush[/]   — Limpia el cache de Redis")
    console.print("  [cyan]/salir[/]   — Salir")
    console.print()

    chat_fn, vectorstore, semantic_cache = build_chat_chain()
    debug_mode = False
    chat_history = []

    collection = vectorstore._collection
    count = collection.count()
    cache_stats = semantic_cache.stats()
    console.print(f"📊 Base de conocimiento: {count} chunks indexados")
    console.print(f"⚡ Cache: {cache_stats['entries']} entradas en Redis")
    console.print(f"🧠 Memoria: últimos {MEMORY_WINDOW} mensajes\n")

    while True:
        try:
            question = console.input("[bold green]Tú:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue

        if question.lower() == "/salir":
            console.print("👋 ¡Hasta luego!")
            break

        if question.lower() == "/debug":
            debug_mode = not debug_mode
            console.print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
            continue

        if question.lower() == "/stats":
            stats = semantic_cache.stats()
            console.print(f"📊 Chunks indexados: {collection.count()}")
            console.print(f"🧠 Modelo embeddings: {EMBEDDING_MODEL}")
            console.print(f"💾 Almacenamiento: {CHROMA_DIR}/")
            console.print(f"💬 Mensajes en memoria: {len(chat_history)}")
            console.print(f"⚡ Cache: {stats['entries']} entradas, threshold={stats['threshold']}")
            continue

        if question.lower() == "/clear":
            chat_history.clear()
            console.print("🗑️  Historial limpiado")
            continue

        if question.lower() == "/flush":
            cleared = semantic_cache.clear()
            console.print(f"🗑️  Cache limpiado ({cleared} entradas eliminadas)")
            continue

        # Ejecutar RAG con historial
        answer, cache_hit = chat_fn(question, chat_history[-MEMORY_WINDOW:])

        # Guardar en memoria
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        console.print()
        status = " ⚡ [yellow](cache)[/]" if cache_hit else ""
        console.print(Panel(
            Markdown(answer),
            title=f"🤖 Asistente{status}",
            border_style="blue",
        ))

        if debug_mode:
            from rag.retriever import get_retriever
            docs = get_retriever().invoke(question)
            console.print("\n[dim]── Chunks recuperados ──[/]")
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "?")
                console.print(f"[dim]  {i+1}. {source}[/]")
                console.print(f"[dim]     {doc.page_content[:80]}...[/]")

        console.print()


if __name__ == "__main__":
    main()
