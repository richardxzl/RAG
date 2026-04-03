# LangChain + LangGraph — RAG Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.2-green)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-purple)](https://langchain-ai.github.io/langgraph)
[![LLM](https://img.shields.io/badge/LLM-Multi--proveedor-orange)](https://python.langchain.com/docs/integrations/chat/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-red)](https://trychroma.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)](https://fastapi.tiangolo.com)

Repositorio de aprendizaje autodidacta para dominar **LangChain** y **LangGraph** desde los fundamentos hasta producción. 65 scripts progresivos, cada uno con su nota de aprendizaje en `learning/`.

---

## Qué aprenderás

```
PARTE 1 — LangChain Profundo          PARTE 2 — LangGraph
─────────────────────────────         ──────────────────────────────────────
Módulo 1: LCEL Avanzado               Módulo 7:  Fundamentos de LangGraph
Módulo 2: Prompts y Output Parsers    Módulo 8:  RAG como Grafo (Corrective RAG)
Módulo 3: Retrievers Avanzados        Módulo 9:  Agentes — ReAct + HITL
Módulo 4: Document Loaders            Módulo 10: Estado Persistente + Time Travel
Módulo 5: Evaluación (RAGAS)          Módulo 11: Multi-Agent Systems
Módulo 6: Observabilidad

PARTE 3 — Producción
─────────────────────────────
Módulo 12: API REST (FastAPI + Streaming + Rate Limiting + Health Checks)
Módulo 13: Supabase pgvector (Migración, Índices, Filtros SQL)
```

---

## Requisitos previos

| Herramienta | Versión mínima | Para qué |
|-------------|---------------|----------|
| Python | 3.10+ | Todo |
| API key del proveedor LLM | — | Según proveedor elegido (ver abajo) |
| Redis | 7+ (opcional) | Módulos de cache (3, 6) |
| PostgreSQL + pgvector | — (opcional) | Módulo 13 — requiere Supabase |

> **Redis es opcional.** Los módulos de cache se degradan automáticamente si no está disponible.
> **Con Ollama no necesitas ninguna API key** — el LLM corre 100% local.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/rags.git
cd rags
```

### 2. Crear el entorno virtual

```bash
python3.10 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt

# Para el Módulo 12 (FastAPI)
pip install fastapi uvicorn[standard] slowapi

# Para el Módulo 13 (pgvector)
pip install langchain-postgres psycopg2-binary
```

> La primera ejecución descarga el modelo de embeddings `all-MiniLM-L6-v2` (~90 MB).

### 4. Elegir proveedor LLM e instalar su paquete

El proyecto soporta cualquier LLM compatible con LangChain. Solo necesitas dos cosas: instalar el paquete del proveedor y configurar la variable `LLM_MODEL` en `.env`.

| Proveedor | Paquete | `LLM_MODEL` | API Key |
|-----------|---------|-------------|---------|
| **Anthropic** (default) | `langchain-anthropic` ✓ ya incluido | `anthropic/claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| **OpenAI** | `pip install langchain-openai` | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| **Ollama** (local, sin API key) | `pip install langchain-ollama` | `ollama/llama3.2` | — |
| **Google Gemini** | `pip install langchain-google-genai` | `google_genai/gemini-1.5-flash` | `GOOGLE_API_KEY` |
| **Groq** | `pip install langchain-groq` | `groq/llama-3.1-8b-instant` | `GROQ_API_KEY` |

### 5. Configurar `.env`

Crea un archivo `.env` en la raíz del proyecto:

```bash
# ── LLM ─────────────────────────────────────────────────────────
# Cambia esta línea para usar un proveedor diferente (ver tabla arriba)
LLM_MODEL=anthropic/claude-haiku-4-5-20251001

# API key del proveedor activo (solo la que necesites)
ANTHROPIC_API_KEY=sk-ant-your-key-here
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=AIza...
# GROQ_API_KEY=gsk_...

# ── Opcionales ───────────────────────────────────────────────────
# Redis (cache semántico)
REDIS_URL=redis://localhost:6379

# Supabase pgvector (Módulo 13)
DATABASE_URL=postgresql://postgres.[ref]:[pass]@aws-0-[region].pooler.supabase.com:6543/postgres

# LangSmith tracing
# LANGCHAIN_API_KEY=your-langsmith-key-here
# LANGCHAIN_TRACING_V2=true
```

### 5. Ingestar documentos (obligatorio — crea el vectorstore)

```bash
# Coloca tus PDFs, TXTs o MDs en ./docs/
python 01_ingest.py
```

Esto crea la carpeta `chroma_db/` con los embeddings locales.

---

## Cambiar de proveedor LLM

Solo hay que editar `LLM_MODEL` en `.env` — ningún script necesita cambios.

### Ejemplo 1: Ollama (local, gratuito, sin API key)

```bash
# 1. Instalar Ollama: https://ollama.com
# 2. Descargar el modelo
ollama pull llama3.2

# 3. Instalar el paquete LangChain
pip install langchain-ollama

# 4. Actualizar .env
LLM_MODEL=ollama/llama3.2
# (eliminar o dejar comentada cualquier API key — no se necesita)

# 5. Ejecutar cualquier script normalmente
python 02_query.py
python 45_loop_reformular.py  # Corrective RAG completo con Llama
```

### Ejemplo 2: OpenAI GPT-4o mini

```bash
# 1. Instalar el paquete
pip install langchain-openai

# 2. Actualizar .env
LLM_MODEL=openai/gpt-4o-mini
OPENAI_API_KEY=sk-...

# 3. Sin más cambios — todo funciona igual
python 02_query.py
python 48_react_pattern.py  # ReAct agent con GPT-4o mini
```

### Cómo funciona internamente

```python
# rag/chain.py — get_llm() usa init_chat_model de LangChain
from langchain.chat_models import init_chat_model

def get_llm():
    return init_chat_model(LLM_MODEL, temperature=LLM_TEMPERATURE)
    # LLM_MODEL viene de .env — "proveedor/modelo"
```

`init_chat_model` detecta el proveedor por el prefijo (`anthropic/`, `openai/`, `ollama/`...) y construye el cliente correcto. Los 65 scripts no saben qué proveedor hay debajo.

---

## Ejecutar los scripts

### Por módulo

```bash
# Módulo 1 — LCEL avanzado
python 04_runnable_lambda.py
python 05_runnable_branch.py
python 06_fallbacks.py

# Módulo 3 — Retrievers avanzados
python 14_similarity_vs_mmr.py
python 16_parent_child_retriever.py
python 17_multi_query_retriever.py

# Módulo 7 — Fundamentos de LangGraph
python 34_state_graph.py          # StateGraph, nodos y edges
python 36_edges.py                # Edges condicionales
python 38_visualizar_grafo.py     # Mermaid diagram en terminal

# Módulo 8 — Corrective RAG
python 39_rag_as_graph.py         # RAG básico como grafo
python 45_loop_reformular.py      # Corrective RAG completo

# Módulo 9 — Agentes
python 46_que_es_agente.py        # Chain vs Agente
python 48_react_pattern.py        # ReAct loop
python 50_human_in_the_loop.py    # Pausar y confirmar acciones

# Módulo 10 — Estado persistente
python 51_checkpointing.py        # MemorySaver, threads independientes
python 54_time_travel.py          # Rollback a estados anteriores

# Módulo 11 — Multi-Agent
python 55_supervisor_pattern.py   # Supervisor que coordina agentes
python 58_shared_vs_isolated_state.py  # Send() fan-out / fan-in

# Módulo 12 — API REST
python 59_fastapi_wrapper.py      # Demo sin servidor
python 60_streaming_endpoint.py   # Streaming directo
python 61_rate_limiting.py        # Rate limiting manual
python 62_health_checks.py        # Liveness + readiness probes

# Módulo 13 — pgvector
python 63_pgvector_migrate.py     # Comparación + SQL de setup
python 64_pgvector_indices.py     # HNSW vs IVFFlat, benchmarks
python 65_pgvector_metadata_filter.py  # Filtros SQL avanzados
```

### Arrancar la API REST

```bash
# Desarrollo — recarga automática
uvicorn 59_fastapi_wrapper:app --reload --port 8000

# Producción — múltiples workers
uvicorn 59_fastapi_wrapper:app --workers 4 --port 8000
```

Docs interactivos: `http://localhost:8000/docs`

### Recomendación de orden

Seguir la numeración: `01_ingest.py` → `02_query.py` → ... → `65_pgvector_metadata_filter.py`. Cada script construye sobre los conceptos del anterior.

---

## Estructura del proyecto

```
rags/
├── rag/                          # Paquete principal reutilizable
│   ├── config.py                 # Configuración centralizada
│   ├── embeddings.py             # Modelo de embeddings local
│   ├── retriever.py              # Vectorstore + retriever
│   ├── chain.py                  # LCEL chains + prompts
│   └── cache.py                  # Semantic cache + Redis
│
├── learning/                     # Notas de aprendizaje (00–65)
│   ├── 00_PROGRAMA.md            # Índice completo del programa
│   ├── 34_state_graph.md
│   ├── 45_loop_reformular.md
│   └── ...
│
├── docs/                         # TUS documentos para el RAG (gitignored)
│   └── [pon aquí tus PDFs, TXTs, MDs]
│
├── 01_ingest.py                  # Ingesta → ChromaDB
├── 02_query.py                   # RAG básico
├── 03_chat.py                    # Chat con memoria
├── 04–33_*.py                    # Parte 1: LangChain Profundo
├── 34–58_*.py                    # Parte 2: LangGraph
├── 59–65_*.py                    # Parte 3: Producción
│
├── requirements.txt
├── ROADMAP.md                    # Estado del progreso
└── learning/00_PROGRAMA.md       # Índice completo con todos los scripts
```

---

## Highlights técnicos

### Corrective RAG (script 45)

```
Pregunta → Router → Retriever → Grader ──(irrelevante)──→ Reformular ──┐
                                    │                                    │
                                    └──(relevante)──→ Generator → Check │
                                                           │             │
                                                   (no fiel) → Regenerar│
                                                           │             │
                                                          END ←──────────┘
```

### Agente ReAct con HITL (script 50)

```python
# El agente pausa antes de ejecutar acciones sensibles
agente = create_react_agent(llm, tools, checkpointer=MemorySaver())

# El grafo se congela en interrupt() y espera aprobación
resultado = grafo.invoke(Command(resume="s"), config=thread_id)
```

### Multi-Agent fan-out / fan-in (script 58)

```python
# Send() despacha N instancias del nodo en paralelo (Map)
def fan_out(estado) -> list[Send]:
    return [Send("agente", {"perspectiva": p}) for p in perspectivas]

# operator.add acumula los resultados (Reduce)
class Estado(TypedDict):
    resultados: Annotated[list[str], operator.add]
```

### API REST con streaming (scripts 59–60)

```python
# Token por token con Server-Sent Events
@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    return StreamingResponse(sse_generator(req.question), media_type="text/event-stream")
```

### Embeddings 100% locales

No se envían documentos a ninguna API de embeddings. Usa `all-MiniLM-L6-v2` via `sentence-transformers`.

---

## Tecnologías

| Categoría | Librería |
|-----------|---------|
| Orquestación | LangChain 1.2, LangGraph 1.1 |
| LLM (cualquiera) | Anthropic, OpenAI, Ollama, Google Gemini, Groq… |
| Vector store | ChromaDB 1.5 (local) / pgvector (Supabase) |
| Embeddings | sentence-transformers (local, sin API) |
| Cache | Redis 7 (Semantic Cache) |
| API | FastAPI + uvicorn |
| Output | Rich (terminal UI) |

---

## Roadmap

Ver [ROADMAP.md](ROADMAP.md) para el estado completo del aprendizaje.

- [x] **PARTE 1** — LangChain Profundo (Módulos 1–6, scripts 01–33)
- [x] **PARTE 2** — LangGraph (Módulos 7–11, scripts 34–58)
- [x] **PARTE 3** — Producción (Módulos 12–13, scripts 59–65)
