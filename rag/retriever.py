"""
retriever.py — Setup del vector store y retriever.
"""
from langchain_chroma import Chroma
from rag.config import COLLECTION_NAME, CHROMA_DIR, RETRIEVAL_K, RETRIEVAL_SEARCH_TYPE
from rag.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    """Carga el vector store de ChromaDB existente."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )


def get_retriever(vectorstore: Chroma = None):
    """Crea un retriever desde el vector store."""
    if vectorstore is None:
        vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type=RETRIEVAL_SEARCH_TYPE,
        search_kwargs={"k": RETRIEVAL_K},
    )
