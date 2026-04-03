"""
chain.py — LCEL chain builders con cache integrado.

Provee funciones para construir chains RAG que usan el cache
de forma transparente. El llamador no necesita saber si la
respuesta vino del cache o del LLM.
"""
import logging
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

from rag.config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from rag.retriever import get_retriever, get_vectorstore
from rag.cache import SemanticCache, RetrievalCache


def format_docs(docs) -> str:
    """Convierte Documents en string separado por delimitadores."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_llm() -> BaseChatModel:
    """
    Crea la instancia del LLM según LLM_MODEL en .env.

    Formato: "proveedor/modelo"
      anthropic/claude-haiku-4-5-20251001  → ANTHROPIC_API_KEY
      openai/gpt-4o-mini                   → OPENAI_API_KEY
      ollama/llama3.2                      → sin API key (local)
      google_genai/gemini-1.5-flash        → GOOGLE_API_KEY
      groq/llama-3.1-8b-instant           → GROQ_API_KEY
    """
    if "/" in LLM_MODEL:
        provider, model = LLM_MODEL.split("/", 1)
    else:
        provider, model = "anthropic", LLM_MODEL

    logger.info("LLM activo: provider=%s model=%s temperature=%s", provider, model, LLM_TEMPERATURE)

    kwargs = {"temperature": LLM_TEMPERATURE}
    # Ollama y otros proveedores locales no aceptan max_tokens como parámetro
    if provider not in ("ollama",):
        kwargs["max_tokens"] = LLM_MAX_TOKENS
    return init_chat_model(model, model_provider=provider, **kwargs)


# ── Prompts ────────────────────────────────────────────────

QUERY_PROMPT = ChatPromptTemplate.from_template(
    """Eres un asistente útil que responde preguntas basándose
ÚNICAMENTE en el contexto proporcionado. Si la respuesta no está en el
contexto, di "No tengo suficiente información para responder eso."

Contexto:
{context}

Pregunta: {question}

Respuesta (en español, clara y concisa):"""
)

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente experto que responde preguntas basándose
en el contexto proporcionado. Si la información no está en el contexto,
dilo claramente. Responde siempre en español.

Contexto relevante:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def build_query_chain():
    """
    Chain para consulta simple (02_query.py).
    Retorna: (query_fn, retriever) donde query_fn acepta una pregunta
    y retorna (answer, source_docs, cache_hit).
    """
    retriever = get_retriever()
    llm = get_llm()
    semantic_cache = SemanticCache()
    retrieval_cache = RetrievalCache()

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["docs"]),
        )
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )

    def query_fn(question: str) -> tuple[str, list, bool]:
        # Capa 1: Semantic cache
        cached_answer = semantic_cache.get(question)
        if cached_answer is not None:
            # Aún necesitamos los docs para mostrar fuentes
            docs = retrieval_cache.get(question)
            if docs is None:
                docs = retriever.invoke(question)
                retrieval_cache.set(question, docs)
            return cached_answer, docs, True

        # Capa 2: Retrieval cache
        docs = retrieval_cache.get(question)
        if docs is None:
            docs = retriever.invoke(question)
            retrieval_cache.set(question, docs)

        # LLM call
        answer = rag_chain.invoke({"question": question, "docs": docs})

        # Guardar en semantic cache
        semantic_cache.set(question, answer)

        return answer, docs, False

    return query_fn, semantic_cache


def build_chat_chain():
    """
    Chain para chat interactivo (03_chat.py).
    Retorna: (chat_fn, vectorstore, semantic_cache) donde chat_fn
    acepta (question, chat_history) y retorna (answer, cache_hit).
    """
    vectorstore = get_vectorstore()
    retriever = get_retriever(vectorstore)
    llm = get_llm()
    semantic_cache = SemanticCache()
    retrieval_cache = RetrievalCache()

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["docs"]),
        )
        | CHAT_PROMPT
        | llm
        | StrOutputParser()
    )

    def chat_fn(question: str, chat_history: list) -> tuple[str, bool]:
        # Capa 1: Semantic cache (solo si no hay historial relevante)
        # Con historial, la misma pregunta puede necesitar respuesta diferente
        if len(chat_history) == 0:
            cached_answer = semantic_cache.get(question)
            if cached_answer is not None:
                return cached_answer, True

        # Capa 2: Retrieval cache
        docs = retrieval_cache.get(question)
        if docs is None:
            docs = retriever.invoke(question)
            retrieval_cache.set(question, docs)

        # LLM call con historial
        answer = rag_chain.invoke({
            "question": question,
            "chat_history": chat_history,
            "docs": docs,
        })

        # Solo cachear si no hay historial (respuesta independiente)
        if len(chat_history) == 0:
            semantic_cache.set(question, answer)

        return answer, False

    return chat_fn, vectorstore, semantic_cache
