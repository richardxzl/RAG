# Programa Completo: LangChain + LangGraph

> Cada módulo se implementa en el proyecto RAG real, con código funcional y debugging.
> No es teoría suelta — cada punto produce un script ejecutable con su nota en `learning/`.

---

## Estado del proyecto

```
PARTE 1 — LangChain Profundo    ✅ completa  (scripts 01–33)
PARTE 2 — LangGraph             ✅ completa  (scripts 34–58)
PARTE 3 — Producción            ✅ completa  (scripts 59–65)
```

Base ya implementada antes de los módulos:
- [x] Ingesta de documentos (PDFs, TXT, MD) → ChromaDB
- [x] Embeddings locales (all-MiniLM-L6-v2)
- [x] LCEL básico (pipe operator, RunnableParallel, StrOutputParser)
- [x] Memoria conversacional (Buffer Window)
- [x] Semantic Cache + Retrieval Cache con Redis
- [x] Modularización (`rag/` package: config, embeddings, retriever, chain, cache)

---

## PARTE 1 — LangChain Profundo

### Módulo 1: LCEL Avanzado ✅
- [x] 1.1 — RunnableLambda → `04_runnable_lambda.py` · `learning/04_runnable_lambda.md`
- [x] 1.2 — RunnableBranch → `05_runnable_branch.py` · `learning/05_runnable_branch.md`
- [x] 1.3 — Fallbacks → `06_fallbacks.py` · `learning/06_fallbacks.md`
- [x] 1.4 — Streaming → `07_streaming.py` · `learning/07_streaming.md`
- [x] 1.5 — Batch → `08_batch.py` · `learning/08_batch.md`

### Módulo 2: Prompts y Output Parsers ✅
- [x] 2.1 — ChatPromptTemplate vs PromptTemplate → `09_prompt_templates.py`
- [x] 2.2 — Few-shot prompting → `10_few_shot.py`
- [x] 2.3 — StructuredOutputParser → `11_structured_output.py`
- [x] 2.4 — PydanticOutputParser → `12_pydantic_output.py`
- [x] 2.5 — Output fixing → `13_output_fixing.py`

### Módulo 3: Retrievers Avanzados ✅
- [x] 3.1 — Similarity vs MMR → `14_similarity_vs_mmr.py`
- [x] 3.2 — Hybrid Search (BM25 + embeddings) → `15_hybrid_search.py`
- [x] 3.3 — Parent-Child Retriever → `16_parent_child_retriever.py`
- [x] 3.4 — Multi-Query Retriever → `17_multi_query_retriever.py`
- [x] 3.5 — Contextual Compression → `18_contextual_compression.py`
- [x] 3.6 — Self-Query Retriever → `19_self_query_retriever.py`

### Módulo 4: Document Loaders y Chunking Avanzado ✅
- [x] 4.1 — Markdown splitter → `20_markdown_splitter.py`
- [x] 4.2 — Semantic chunking → `21_semantic_chunking.py`
- [x] 4.3 — Metadata enrichment → `22_metadata_enrichment.py`
- [x] 4.4 — Comparar chunk sizes → `23_chunk_size_comparison.py`

### Módulo 5: Evaluación Automatizada ✅
- [x] 5.1 — Dataset de evaluación → `24_eval_dataset.py`
- [x] 5.2 — Métricas manuales → `25_metricas_manuales.py`
- [x] 5.3 — RAGAS end-to-end → `26_ragas_eval.py`
- [x] 5.4 — Comparar configuraciones → `27_comparar_configuraciones.py`
- [x] 5.5 — Regression testing → `28_regression_testing.py`

### Módulo 6: Observabilidad y Debugging ✅
- [x] 6.1 — Logging estructurado → `29_logging_estructurado.py`
- [x] 6.2 — Callbacks de LangChain → `30_callbacks.py`
- [x] 6.3 — Tracing (LangSmith / local) → `31_tracing.py`
- [x] 6.4 — Métricas de latencia → `32_latencia_por_componente.py`
- [x] 6.5 — Error handling / graceful degradation → `33_error_handling.py`

---

## PARTE 2 — LangGraph

### Módulo 7: Fundamentos de LangGraph ✅
- [x] 7.1 — StateGraph: estado, nodos y edges → `34_state_graph.py`
- [x] 7.2 — Nodes: funciones que transforman el estado → `35_nodes.py`
- [x] 7.3 — Edges: condicionales y rutas → `36_edges.py`
- [x] 7.4 — Compilar y ejecutar un grafo mínimo → `37_compilar_grafo.py`
- [x] 7.5 — Visualizar el grafo (Mermaid) → `38_visualizar_grafo.py`

### Módulo 8: RAG como Grafo ✅
- [x] 8.1 — Migrar RAG de LCEL a LangGraph → `39_rag_as_graph.py`
- [x] 8.2 — Nodo Router → `40_nodo_router.py`
- [x] 8.3 — Nodo Retriever → `41_nodo_retriever.py`
- [x] 8.4 — Nodo Grader → `42_nodo_grader.py`
- [x] 8.5 — Nodo Generator → `43_nodo_generator.py`
- [x] 8.6 — Nodo Hallucination Check → `44_hallucination_check.py`
- [x] 8.7 — Loop: reformular → reintentar (Corrective RAG) → `45_loop_reformular.py`

### Módulo 9: Agentes con LangGraph ✅
- [x] 9.1 — ¿Qué es un agente? → `46_que_es_agente.py`
- [x] 9.2 — Tool calling → `47_tool_calling.py`
- [x] 9.3 — ReAct pattern → `48_react_pattern.py`
- [x] 9.4 — Tools custom (RAG, web, cálculos) → `49_tools_custom.py`
- [x] 9.5 — Human-in-the-loop → `50_human_in_the_loop.py`

### Módulo 10: Estado Persistente y Checkpoints ✅
- [x] 10.1 — Checkpointing → `51_checkpointing.py`
- [x] 10.2 — Memory as state → `52_memory_as_state.py`
- [x] 10.3 — Persistent threads → `53_persistent_threads.py`
- [x] 10.4 — Time travel → `54_time_travel.py`

### Módulo 11: Multi-Agent Systems ✅
- [x] 11.1 — Supervisor pattern → `55_supervisor_pattern.py`
- [x] 11.2 — Subgraphs → `56_subgraphs.py`
- [x] 11.3 — Handoff → `57_handoff.py`
- [x] 11.4 — Shared state vs isolated state → `58_shared_vs_isolated_state.py`

---

## PARTE 3 — Producción ✅

### Módulo 12: API REST ✅
- [x] 12.1 — FastAPI wrapper del RAG → `59_fastapi_wrapper.py` · `learning/59_fastapi_wrapper.md`
- [x] 12.2 — Streaming endpoint (SSE) → `60_streaming_endpoint.py` · `learning/60_streaming_endpoint.md`
- [x] 12.3 — Rate limiting → `61_rate_limiting.py` · `learning/61_rate_limiting.md`
- [x] 12.4 — Health checks → `62_health_checks.py` · `learning/62_health_checks.md`

### Módulo 13: Supabase pgvector ✅
- [x] 13.1 — Migrar de ChromaDB a pgvector → `63_pgvector_migrate.py` · `learning/63_pgvector_migrate.md`
- [x] 13.2 — Índices y performance → `64_pgvector_indices.py` · `learning/64_pgvector_indices.md`
- [x] 13.3 — Filtrado por metadata en SQL → `65_pgvector_metadata_filter.py` · `learning/65_pgvector_metadata_filter.md`

---

## Orden de lectura recomendado

```
01_ingest → 02_query → 03_chat
    ↓
04–08 (LCEL avanzado)
    ↓
09–13 (Prompts y Parsers)
    ↓
14–19 (Retrievers)
    ↓
20–23 (Chunking)
    ↓
24–28 (Evaluación)     ←── paralelizable con 29–33
    ↓
29–33 (Observabilidad)
    ↓
34–38 (LangGraph fundamentos)
    ↓
39–45 (RAG como Grafo — Corrective RAG)
    ↓
46–50 (Agentes — ReAct + HITL)
    ↓
51–54 (Estado persistente — checkpointing + time travel)
    ↓
55–58 (Multi-Agent — supervisor, subgraphs, handoff, Send)
```
