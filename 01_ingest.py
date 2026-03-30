"""
01_ingest.py — Ingesta de documentos al vector store
Carga PDFs y textos de ./docs, los fragmenta y almacena en ChromaDB
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()

# ── Configuración ──────────────────────────────────────────
DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "mi_knowledge_base"

CHUNK_SIZE = 1000       # Caracteres por chunk
CHUNK_OVERLAP = 200     # Solapamiento
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modelo local gratuito


def load_documents(docs_dir: str):
    """Carga todos los PDFs y TXTs del directorio"""
    documents = []
    docs_path = Path(docs_dir)

    for file_path in docs_path.glob("**/*"):
        if file_path.suffix.lower() == ".pdf":
            console.print(f"  📄 Cargando PDF: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        elif file_path.suffix.lower() in [".txt", ".md"]:
            console.print(f"  📝 Cargando texto: {file_path.name}")
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

    return documents


def chunk_documents(documents):
    """Fragmenta documentos en chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def create_vectorstore(chunks):
    """Genera embeddings y almacena en ChromaDB"""
    console.print(f"\n🧠 Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    console.print("💾 Creando vector store en ChromaDB...")

    # Eliminar DB anterior si existe (para re-ingesta limpia)
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore


def main():
    console.rule("[bold blue]RAG Lab — Ingesta de Documentos")

    # 1. Cargar documentos
    console.print("\n📂 Cargando documentos desde ./docs/")
    documents = load_documents(DOCS_DIR)

    if not documents:
        console.print("[red]❌ No se encontraron documentos en ./docs/")
        console.print("   Pon algunos PDFs o TXTs ahí y vuelve a ejecutar.")
        return

    console.print(f"   ✅ {len(documents)} páginas/archivos cargados")

    # 2. Chunking
    console.print(f"\n✂️  Fragmentando (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_documents(documents)
    console.print(f"   ✅ {len(chunks)} chunks generados")

    # 3. Mostrar preview de chunks
    table = Table(title="Preview de chunks (primeros 3)")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Fuente", style="green", width=25)
    table.add_column("Contenido (primeros 100 chars)", style="white")

    for i, chunk in enumerate(chunks[:3]):
        source = chunk.metadata.get("source", "?")
        table.add_row(
            str(i + 1),
            Path(source).name,
            chunk.page_content[:100] + "..."
        )
    console.print(table)

    # 4. Embeddings + Vector Store
    start = time.time()
    vectorstore = create_vectorstore(chunks)
    elapsed = time.time() - start

    console.print(f"\n✅ [bold green]Ingesta completada en {elapsed:.1f}s")
    console.print(f"   → {len(chunks)} chunks almacenados en {CHROMA_DIR}/")
    console.print(f"   → Modelo de embeddings: {EMBEDDING_MODEL}")
    console.print(f"   → Dimensiones: 384")

    # 5. Test rápido de búsqueda
    console.rule("[bold yellow]Test rápido de retrieval")
    test_query = "¿De qué trata este documento?"
    results = vectorstore.similarity_search_with_score(test_query, k=3)

    for i, (doc, score) in enumerate(results):
        console.print(f"\n  [cyan]Resultado {i+1}[/] (similitud: {1-score:.3f})")
        console.print(f"  Fuente: {Path(doc.metadata.get('source', '?')).name}")
        console.print(f"  {doc.page_content[:150]}...")


if __name__ == "__main__":
    main()
