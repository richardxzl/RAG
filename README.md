# LangChain + LangGraph — RAG Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.2-green)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-purple)](https://langchain-ai.github.io/langgraph)
[![Claude](https://img.shields.io/badge/LLM-Claude%20Haiku-orange)](https://anthropic.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-red)](https://trychroma.com)

Repositorio de aprendizaje autodidacta para dominar **LangChain** y **LangGraph** desde los fundamentos hasta sistemas de agentes con estado. 50 scripts progresivos, cada uno con su nota de aprendizaje en `docs/learning/`.

---

## Qué aprenderás

```
PARTE 1 — LangChain Profundo          PARTE 2 — LangGraph
─────────────────────────────         ─────────────────────────────────────
Módulo 1: LCEL Avanzado               Módulo 7: Fundamentos de LangGraph
Módulo 2: Prompts y Output Parsers    Módulo 8: RAG como Grafo (Corrective RAG)
Módulo 3: Retrievers Avanzados        Módulo 9: Agentes — ReAct + HITL
Módulo 4: Document Loaders
Módulo 5: Evaluación (RAGAS)
Módulo 6: Observabilidad
```

---

## Requisitos previos

| Herramienta | Versión mínima | Para qué |
|-------------|---------------|----------|
| Python | 3.10+ | Todo |
| API Key Anthropic | — | Llamadas al LLM |
| Redis | 7+ (opcional) | Módulos de cache (3, 6) |

> **Redis es opcional.** Los módulos de cache se degradan automáticamente si Redis no está disponible.

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
```

> La primera ejecución descarga el modelo de embeddings `all-MiniLM-L6-v2` (~90 MB).

### 4. Configurar credenciales

Crea un archivo `.env` en la raíz del proyecto:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Redis (opcional)
REDIS_URL=redis://localhost:6379

# LangSmith tracing (opcional)
# LANGCHAIN_API_KEY=your-langsmith-key-here
# LANGCHAIN_TRACING_V2=true
```

Obtén tu API key en [console.anthropic.com](https://console.anthropic.com/).

### 5. Ingestar documentos (obligatorio — crea el vectorstore)

```bash
# Coloca tus PDFs, TXTs o MDs en ./docs/
python 01_ingest.py
```

Esto crea la carpeta `chroma_db/` con los embeddings locales.

---

## Ejecutar los scripts

### Por módulo

```bash
# Módulo 1 — LCEL básico
python 04_runnable_lambda.py
python 05_runnable_branch.py
python 06_fallbacks.py
python 07_streaming.py

# Módulo 3 — Retrievers avanzados
python 14_similarity_vs_mmr.py
python 16_parent_child_retriever.py
python 17_multi_query_retriever.py

# Módulo 7 — Fundamentos de LangGraph
python 34_state_graph.py         # StateGraph, nodos y edges
python 36_edges.py               # Edges condicionales
python 38_visualizar_grafo.py    # Mermaid diagram en terminal

# Módulo 8 — Corrective RAG
python 39_rag_as_graph.py        # RAG básico como grafo
python 45_loop_reformular.py     # Corrective RAG completo

# Módulo 9 — Agentes
python 46_que_es_agente.py       # Chain vs Agente
python 48_react_pattern.py       # ReAct loop
python 50_human_in_the_loop.py   # Pausar y confirmar acciones
```

### Recomendación de orden

Seguir la numeración: `01_ingest.py` → `02_query.py` → ... → `50_human_in_the_loop.py`. Cada script construye sobre los conceptos del anterior.

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
├── learning/                     # Notas de aprendizaje (01–50)
│   ├── 34_state_graph.md
│   ├── 45_loop_reformular.md
│   └── ...
│
├── docs/                         # TUS documentos para el RAG (gitignored)
│   └── [pon aquí tus PDFs, TXTs, MDs]
│
├── 01_ingest.py                  # Ingesta de documentos → ChromaDB
├── 02_query.py                   # RAG básico
├── 03_chat.py                    # Chat con memoria
├── 04_runnable_lambda.py         # M1: LCEL
│   ...
├── 34_state_graph.py             # M7: LangGraph fundamentos
│   ...
├── 45_loop_reformular.py         # M8: Corrective RAG completo
│   ...
├── 50_human_in_the_loop.py       # M9: HITL con interrupt()
│
├── requirements.txt
├── ROADMAP.md                    # Progreso del aprendizaje
└── rag-guia-completa.md          # Guía de referencia rápida
```

---

## Highlights técnicos

### Corrective RAG (script 45)

El sistema RAG más completo del repositorio. Implementa el patrón de auto-corrección:

```
Pregunta → Router → Retriever → Grader ──(no relevante)──→ Reformular ──┐
                                    │                                     │
                                    └──(relevante)──→ Generator → Check  │
                                                           │              │
                                                   (no fiel) → Regenerar │
                                                           │              │
                                                          END ←───────────┘
```

### Agente ReAct con HITL (script 50)

```python
# El agente pausa antes de ejecutar acciones sensibles
agente = create_react_agent(llm, tools, checkpointer=MemorySaver())

# El grafo se congela en interrupt() y espera tu aprobación
resultado = grafo.invoke(Command(resume="s"), config=thread_id)
```

### Embeddings 100% locales

No se envían los documentos a ninguna API de embeddings. Usa `all-MiniLM-L6-v2` via `sentence-transformers`.

---

## Tecnologías

| Categoría | Librería |
|-----------|---------|
| Orquestación | LangChain 1.2, LangGraph 1.1 |
| LLM | Claude Haiku 4.5 (Anthropic) |
| Vector store | ChromaDB 1.5 (local) |
| Embeddings | sentence-transformers (local, sin API) |
| Cache | Redis 7 (Semantic Cache) |
| Output | Rich (terminal UI) |

---

## Roadmap

Ver [ROADMAP.md](ROADMAP.md) para el estado completo del aprendizaje.

- [x] **PARTE 1** — LangChain Profundo (Módulos 1–6, scripts 01–33)
- [x] **PARTE 2** — LangGraph (Módulos 7–9, scripts 34–50)
- [ ] Módulo 10: Estado Persistente y Checkpoints
- [ ] Módulo 11: Multi-Agent Systems
- [ ] **PARTE 3** — Producción (FastAPI, pgvector)
