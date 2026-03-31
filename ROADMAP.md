# Roadmap: LangChain + LangGraph

## PARTE 1 — LangChain Profundo

### Módulo 1: LCEL Avanzado
- [x] 1.0 — LCEL básico (pipes, RunnableParallel, StrOutputParser)
- [x] 1.1 — RunnableLambda: funciones custom como pasos del pipeline
- [x] 1.2 — RunnableBranch: condicionales (pregunta X → flujo A, si no → flujo B)
- [x] 1.3 — Fallbacks: retry y modelo alternativo cuando falla un paso
- [x] 1.4 — Streaming: respuestas token por token en tiempo real
- [x] 1.5 — Batch: procesar múltiples preguntas en paralelo

### Módulo 2: Prompts y Output Parsers
- [x] 2.1 — ChatPromptTemplate vs PromptTemplate
- [x] 2.2 — Few-shot prompting: enseñar con ejemplos en el prompt
- [x] 2.3 — StructuredOutputParser: forzar respuestas JSON
- [x] 2.4 — PydanticOutputParser: validación tipada de la salida
- [x] 2.5 — Output fixing: qué hacer cuando el LLM no respeta el formato

### Módulo 3: Retrievers Avanzados
- [x] 3.1 — Similarity vs MMR (Maximum Marginal Relevance)
- [x] 3.2 — Hybrid Search: BM25 + embeddings combinados
- [x] 3.3 — Parent-Child Retriever: chunks pequeños buscan, padres grandes dan contexto
- [x] 3.4 — Multi-Query Retriever: variaciones automáticas de la pregunta
- [x] 3.5 — Contextual Compression: filtrar chunks irrelevantes post-retrieval
- [x] 3.6 — Self-Query Retriever: el LLM genera filtros de metadata

### Módulo 4: Document Loaders y Chunking Avanzado
- [x] 4.1 — Markdown splitter: respetar headers y estructura
- [x] 4.2 — Semantic chunking: cortar por cambio de tema
- [x] 4.3 — Metadata enrichment: agregar metadata útil a cada chunk
- [x] 4.4 — Comparar chunk sizes: test automatizado 500 vs 1000 vs 1500

### Módulo 5: Evaluación Automatizada
- [x] 5.1 — Crear dataset de evaluación (preguntas + ground truth)
- [x] 5.2 — Métricas manuales: precision, recall, faithfulness
- [x] 5.3 — RAGAS: evaluación automatizada end-to-end
- [x] 5.4 — Comparar configuraciones: chunk_size × retriever × k
- [x] 5.5 — Regression testing: detectar si un cambio empeoró la calidad

### Módulo 6: Observabilidad y Debugging
- [x] 6.1 — Logging estructurado por paso del pipeline
- [x] 6.2 — Callbacks de LangChain (on_llm_start, on_retriever_end, etc.)
- [x] 6.3 — Tracing con LangSmith o alternativa local
- [x] 6.4 — Métricas de latencia por componente
- [x] 6.5 — Error handling: graceful degradation (Redis down, API key inválida)

---

## PARTE 2 — LangGraph

### Módulo 7: Fundamentos de LangGraph
- [x] 7.1 — StateGraph: estado, nodos y edges
- [x] 7.2 — Nodes: funciones que transforman el estado
- [x] 7.3 — Edges: condicionales y rutas entre nodos
- [x] 7.4 — Compilar y ejecutar un grafo mínimo
- [x] 7.5 — Visualizar el grafo (mermaid diagram)

### Módulo 8: RAG como Grafo
- [x] 8.1 — Migrar el RAG actual de LCEL a LangGraph
- [x] 8.2 — Nodo Router: ¿necesita RAG o es conversacional?
- [x] 8.3 — Nodo Retriever: busca chunks
- [x] 8.4 — Nodo Grader: ¿los chunks son relevantes? Si no → reformular
- [x] 8.5 — Nodo Generator: genera respuesta con contexto
- [x] 8.6 — Nodo Hallucination Check: ¿la respuesta es fiel a los chunks?
- [x] 8.7 — Loop: reformular → reintentar (max 2 veces)

### Módulo 9: Agentes con LangGraph
- [x] 9.1 — ¿Qué es un agente? (decide qué hacer, no solo ejecuta)
- [x] 9.2 — Tool calling: el LLM decide qué herramientas usar
- [x] 9.3 — ReAct pattern: Reasoning + Acting en loop
- [x] 9.4 — Crear tools custom (RAG, web, cálculos)
- [x] 9.5 — Human-in-the-loop: pausar y pedir confirmación

### Módulo 10: Estado Persistente y Checkpoints
- [x] 10.1 — Checkpointing: guardar estado del grafo entre ejecuciones
- [x] 10.2 — Memory as state: conversación como parte del estado
- [x] 10.3 — Persistent threads: retomar conversaciones
- [x] 10.4 — Time travel: volver a un estado anterior

### Módulo 11: Multi-Agent Systems
- [x] 11.1 — Supervisor pattern: un agente coordina a otros
- [x] 11.2 — Subgraphs: grafos dentro de grafos
- [x] 11.3 — Handoff: un agente pasa el control a otro
- [x] 11.4 — Shared state vs isolated state

---

## PARTE 3 — Producción

### Módulo 12: API REST
- [x] 12.1 — FastAPI wrapper del RAG
- [x] 12.2 — Streaming endpoint (SSE)
- [x] 12.3 — Rate limiting
- [x] 12.4 — Health checks

### Módulo 13: Supabase pgvector
- [x] 13.1 — Migrar de ChromaDB a pgvector
- [x] 13.2 — Índices y performance
- [x] 13.3 — Filtrado por metadata en SQL

---

## Completado previamente
- [x] Ingesta de documentos (PDFs, TXT, MD) → ChromaDB
- [x] Embeddings locales (all-MiniLM-L6-v2)
- [x] Memoria conversacional (Buffer Window)
- [x] Semantic Cache + Retrieval Cache con Redis
- [x] Modularización (rag/ package con config, embeddings, retriever, cache, chain)
