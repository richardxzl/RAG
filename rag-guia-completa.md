# Guía Completa de RAG (Retrieval-Augmented Generation)

> Guía autodidacta paso a paso: conceptos, instalación, configuración, implementación y pruebas.

---

## ¿Qué es RAG?

RAG es un patrón donde en lugar de confiar solo en el conocimiento interno del LLM, le inyectas contexto relevante recuperado de tus propios datos justo antes de que genere la respuesta. El flujo es:

```
Usuario pregunta → Buscas documentos relevantes → Los metes en el prompt → El LLM responde con ese contexto
```

Esto resuelve tres problemas grandes:

- **Alucinaciones** — El modelo se ciñe a datos reales que tú le proporcionas.
- **Datos desactualizados** — Tus documentos siempre están al día.
- **Conocimiento de dominio** — El modelo sabe de TU negocio, no solo de lo que aprendió en el entrenamiento.

---

## Arquitectura de un sistema RAG

Los componentes clave son:

1. **Document Loader** — Ingesta de documentos (PDFs, Markdown, HTML, bases de datos, APIs).
2. **Chunking / Splitting** — Partir los documentos en fragmentos manejables (chunks). Una de las decisiones más críticas: el tamaño y la estrategia de split.
3. **Embedding Model** — Conviertes cada chunk en un vector numérico (embedding) que captura su significado semántico.
4. **Vector Store** — Base de datos que almacena esos embeddings y permite buscar por similitud semántica (no keyword, sino significado).
5. **Retriever** — Dado un query del usuario, lo convierte a embedding y busca los chunks más similares.
6. **LLM + Prompt** — Con los chunks recuperados, armas un prompt enriquecido y el LLM genera la respuesta.

---

## Paso 1: Entorno y herramientas base

### Opción A — Python (más ecosistema RAG, recomendado para aprender)

```bash
# Entorno virtual
python -m venv rag-env
source rag-env/bin/activate  # Linux/Mac

# Paquetes core
pip install langchain langchain-community langchain-anthropic
pip install chromadb          # Vector store local (el más fácil para empezar)
pip install sentence-transformers  # Embeddings locales gratuitos
pip install pypdf             # Para cargar PDFs
pip install python-dotenv     # Variables de entorno
```

### Opción B — TypeScript/Node

```bash
mkdir rag-project && cd rag-project
npm init -y
npm install langchain @langchain/anthropic @langchain/community
npm install chromadb chromadb-default-embed
npm install pdf-parse dotenv
```

### Variables de entorno (.env)

```env
ANTHROPIC_API_KEY=sk-ant-...
# Opcional si usas OpenAI para embeddings:
# OPENAI_API_KEY=sk-...
```

---

## Paso 2: Embeddings — Opciones y tradeoffs

Los embeddings son el corazón de RAG. Tres caminos:

### Locales/gratuitos (para empezar)

- `all-MiniLM-L6-v2` — Rápido, ligero, 384 dimensiones. Perfecto para prototipar.
- `nomic-embed-text` — Más potente, 768 dims, open source.

### API de pago (para producción)

- `text-embedding-3-small` de OpenAI — Barato, buen rendimiento.
- `voyage-3` de Voyage AI — Muy bueno para RAG específicamente.
- `cohere embed-v3` — Buen balance.

### Recomendación

Empieza con `all-MiniLM-L6-v2` local (gratis, sin API key extra) y cuando vayas a producción evalúa Voyage o OpenAI embeddings.

---

## Paso 3: Vector Store — Dónde guardar los embeddings

### Para desarrollo/aprender

- **ChromaDB** — Se instala con pip/npm, corre in-memory o persistido a disco. Cero configuración.

### Para producción

- **Supabase pgvector** — PostgreSQL con extensión `vector` para búsqueda semántica. Datos relacionales + vectores en la misma DB.
- **Pinecone** — Managed, escala fácil, de pago.
- **Qdrant** — Open source, self-hosteable, muy buen rendimiento.
- **Weaviate** — Similar a Qdrant, con buenas features de hybrid search.

---

## Paso 4: Proyecto funcional paso a paso

### Estructura del proyecto

```
rag-lab/
├── .env                  # API keys
├── .venv/                # Entorno virtual
├── docs/                 # Aquí metes tus PDFs/textos
│   └── (pon aquí 1-2 PDFs para probar)
├── chroma_db/            # Se crea solo, almacena vectores
├── 01_ingest.py          # Script de ingesta
├── 02_query.py           # Script de consulta
├── 03_chat.py            # Chat interactivo
└── requirements.txt
```

### Crear el proyecto

```bash
mkdir rag-lab && cd rag-lab

python3 -m venv .venv
source .venv/bin/activate

pip install langchain langchain-anthropic langchain-community langchain-chroma
pip install chromadb
pip install sentence-transformers
pip install pypdf
pip install python-dotenv
pip install rich

pip freeze > requirements.txt

mkdir docs
```

---

### 01_ingest.py — Ingesta de documentos al vector store

Carga PDFs y textos de `./docs`, los fragmenta y almacena en ChromaDB.

```python
"""
01_ingest.py — Ingesta de documentos al vector store
Carga PDFs y textos de ./docs, los fragmenta y almacena en ChromaDB
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
        model_kwargs={"device": "cpu"},      # Usa "cuda" si tienes GPU
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
```

---

### 02_query.py — Consulta RAG (Retrieval + Claude)

```python
"""
02_query.py — Consulta RAG: Retrieval + Claude
"""
import sys
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()
console = Console()

# ── Configuración ──────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "mi_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Prompt personalizado para RAG
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Eres un asistente útil que responde preguntas basándose
ÚNICAMENTE en el contexto proporcionado. Si la respuesta no está en el
contexto, di "No tengo suficiente información para responder eso."

Contexto:
{context}

Pregunta: {question}

Respuesta (en español, clara y concisa):"""
)


def setup_chain():
    """Configura la cadena RAG completa"""
    # Embeddings (mismo modelo que la ingesta)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Cargar vector store existente
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=1024,
    )

    # Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    return chain


def query(chain, question: str):
    """Ejecuta una consulta RAG"""
    console.print(f"\n🔍 Pregunta: [bold]{question}[/]")
    console.print("   Buscando en la base de conocimiento...")

    result = chain.invoke({"query": question})

    # Mostrar respuesta
    console.print(Panel(
        Markdown(result["result"]),
        title="💬 Respuesta",
        border_style="green",
    ))

    # Mostrar fuentes
    console.print("\n📚 [bold]Fuentes utilizadas:[/]")
    for i, doc in enumerate(result["source_documents"]):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        console.print(f"   {i+1}. {source} (pág. {page})")
        console.print(f"      [dim]{doc.page_content[:100]}...[/]")

    return result


def main():
    console.rule("[bold blue]RAG Lab — Consulta")

    chain = setup_chain()

    if len(sys.argv) > 1:
        # Pregunta pasada como argumento
        question = " ".join(sys.argv[1:])
        query(chain, question)
    else:
        # Pregunta por defecto
        query(chain, "¿De qué trata el documento?")
        console.print("\n💡 Tip: python 02_query.py '¿tu pregunta aquí?'")


if __name__ == "__main__":
    main()
```

---

### 03_chat.py — Chat RAG interactivo

```python
"""
03_chat.py — Chat RAG interactivo
"""
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()
console = Console()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "mi_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Eres un asistente experto que responde preguntas basándose
en el contexto proporcionado. Si la información no está en el contexto,
dilo claramente. Responde siempre en español.

Contexto relevante:
{context}

Pregunta del usuario: {question}

Respuesta:"""
)


def setup():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=1024,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    return chain, vectorstore


def show_debug(result):
    """Muestra los chunks recuperados (debug)"""
    console.print("\n[dim]── Chunks recuperados ──[/]")
    for i, doc in enumerate(result["source_documents"]):
        source = doc.metadata.get("source", "?")
        console.print(f"[dim]  {i+1}. {source}[/]")
        console.print(f"[dim]     {doc.page_content[:80]}...[/]")


def main():
    console.rule("[bold blue]🤖 RAG Chat Interactivo")
    console.print("Escribe tus preguntas. Comandos especiales:")
    console.print("  [cyan]/debug[/]   — Activa/desactiva ver chunks recuperados")
    console.print("  [cyan]/stats[/]   — Muestra estadísticas del vector store")
    console.print("  [cyan]/salir[/]   — Salir")
    console.print()

    chain, vectorstore = setup()
    debug_mode = False

    collection = vectorstore._collection
    count = collection.count()
    console.print(f"📊 Base de conocimiento: {count} chunks indexados\n")

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
            console.print(f"📊 Chunks indexados: {collection.count()}")
            console.print(f"🧠 Modelo embeddings: {EMBEDDING_MODEL}")
            console.print(f"💾 Almacenamiento: {CHROMA_DIR}/")
            continue

        # Ejecutar RAG
        result = chain.invoke({"query": question})

        console.print()
        console.print(Panel(
            Markdown(result["result"]),
            title="🤖 Asistente",
            border_style="blue",
        ))

        if debug_mode:
            show_debug(result)

        console.print()


if __name__ == "__main__":
    main()
```

---

## Cómo ejecutar todo

```bash
# 1. Pon documentos en ./docs/
#    (cualquier PDF o .txt que quieras consultar)

# 2. Ingesta
python 01_ingest.py

# 3. Consulta rápida
python 02_query.py "¿Cuáles son los puntos principales?"

# 4. Chat interactivo
python 03_chat.py
```

---

## Chunking — La decisión más crítica

El chunking determina la calidad de tu RAG más que casi cualquier otro factor.

### RecursiveCharacterTextSplitter

El default. Intenta separar por `\n\n`, luego `\n`, luego `. `, etc.

### Parámetros clave

- `chunk_size`: 500-1500 caracteres es el rango típico. Más pequeño = más preciso pero menos contexto. Más grande = más contexto pero puede meter ruido.
- `chunk_overlap`: 10-20% del chunk_size. Evita que una idea quede cortada en la frontera.

### Estrategias avanzadas

- **Semantic Chunking** — Usa embeddings para detectar cambios de tema y cortar ahí.
- **Parent-Child** — Chunks pequeños para buscar, pero recuperas el chunk padre (más grande) para dar contexto al LLM.
- **Markdown/HTML splitter** — Si tus docs tienen estructura, respétala al cortar.

---

## Cómo probar y evaluar tu RAG

### Métricas clave

- **Retrieval Precision** — ¿Los chunks recuperados son relevantes a la pregunta?
- **Retrieval Recall** — ¿Se recuperaron todos los chunks que deberían haberse recuperado?
- **Answer Faithfulness** — ¿La respuesta del LLM es fiel a los chunks (no alucina)?
- **Answer Relevancy** — ¿La respuesta realmente contesta la pregunta?

### Herramientas de evaluación

- **RAGAS** (`pip install ragas`) — Framework de evaluación automática para RAG.
- **LangSmith** — Tracing de LangChain, ves exactamente qué chunks se recuperaron y qué prompt se armó.

### Test manual rápido

```python
# Verifica qué chunks se recuperan para una pregunta
query = "¿Cuál es la política de devoluciones?"
docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:200])
    print(f"Metadata: {doc.metadata}")
```

> Si los chunks no son relevantes, tu problema está en el chunking o en los embeddings, no en el LLM.

---

## Patrones avanzados

### Hybrid Search

Combinar búsqueda semántica (embeddings) con búsqueda por keywords (BM25).

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks, k=4)
semantic = vectorstore.as_retriever(search_kwargs={"k": 4})

hybrid = EnsembleRetriever(
    retrievers=[bm25, semantic],
    weights=[0.4, 0.6]  # Más peso a semántico
)
```

### Re-ranking

Después de recuperar N chunks, un modelo de re-ranking los reordena por relevancia real. Cohere Rerank o modelos cross-encoder son populares.

### Query Transformation

Antes de buscar, transformas la pregunta del usuario: descomponerla en sub-preguntas, expandirla, o reformularla para mejorar la retrieval.

### Multi-Query RAG

Generas 3-5 variaciones de la pregunta original, buscas con cada una, y unes los resultados.

---

## RAG en producción con Supabase pgvector

### Setup SQL en Supabase

```sql
-- Activar la extensión
create extension if not exists vector;

-- Tabla de documentos
create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(384)  -- dimensión según tu modelo
);

-- Índice para búsqueda eficiente
create index on documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Función de búsqueda
create or replace function match_documents(
  query_embedding vector(384),
  match_count int default 5,
  filter jsonb default '{}'
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) as similarity
  from documents d
  where d.metadata @> filter
  order by d.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

---

## Qué experimentar una vez que funcione

1. **Juega con el chunk_size:** Cambia de 1000 a 500, luego a 1500. Haz la misma pregunta y compara qué chunks se recuperan con `/debug`.
2. **Cambia el `k`** (número de chunks recuperados): De 4 a 2, luego a 8. Observa cómo cambia la respuesta.
3. **Prueba con diferentes documentos:** Un PDF técnico, un contrato, unas FAQs. Cada tipo de documento se comporta diferente con el chunking.
4. **Revisa el prompt RAG:** Cambia las instrucciones del `RAG_PROMPT` y observa cómo cambia el comportamiento del LLM.

---

## Roadmap sugerido

| Semana | Objetivo |
|--------|----------|
| 1 | Montar ejemplo básico con ChromaDB + PDF + Claude. Jugar con chunk sizes. |
| 2 | Probar diferentes embeddings (local vs API). Implementar hybrid search. Medir con RAGAS. |
| 3 | Migrar a Supabase pgvector. Construir API REST que reciba preguntas y devuelva respuestas con fuentes. |
| 4 | Integrar en un caso real (ej: knowledge base del dental SaaS para el agente WhatsApp). |

---

## Cómo usar esta guía con un agente en VS Code

Si quieres pasarle este contexto a Claude Code o a otro agente en VS Code, tienes estas opciones:

### Opción 1: CLAUDE.md (para Claude Code)

Crea un archivo `CLAUDE.md` en la raíz de tu proyecto `rag-lab/`:

```markdown
# Contexto del proyecto

Este es un proyecto de aprendizaje de RAG. El stack es:
- Python + LangChain + ChromaDB + Claude (Anthropic)
- Embeddings locales con all-MiniLM-L6-v2
- Vector store: ChromaDB (dev), Supabase pgvector (prod)

## Estructura
- 01_ingest.py: Carga docs de ./docs, chunking, embeddings, almacena en ChromaDB
- 02_query.py: Consulta RAG con retrieval + Claude
- 03_chat.py: Chat interactivo en terminal

## Convenciones
- Usar siempre el mismo EMBEDDING_MODEL en ingesta y consulta
- Chunks: RecursiveCharacterTextSplitter, chunk_size=1000, overlap=200
- Respuestas siempre en español
```

### Opción 2: Archivo de contexto en el proyecto

Pon este mismo `.md` en tu proyecto como `docs/RAG_GUIDE.md` y referéncialo:

```bash
# En Claude Code
claude "lee docs/RAG_GUIDE.md y ayúdame a implementar hybrid search"
```

### Opción 3: Custom instructions en VS Code

Si usas Copilot o similar, copia las secciones relevantes en tu archivo `.github/copilot-instructions.md`.
