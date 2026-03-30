# Multi-Query Retriever — Variaciones automáticas del query

## El problema del embedding frágil

Un embedding captura UNA perspectiva semántica del query. Si el chunk relevante está formulado de forma diferente a la pregunta, el coseno de similitud puede ser bajo aunque la respuesta esté ahí.

```
Query del usuario: "¿Qué ventajas tiene LCEL?"
Chunk relevante:   "Los beneficios de usar LangChain Expression Language son..."

Similitud coseno: 0.62 (quizás por debajo del threshold)
→ El chunk no se recupera
→ El LLM dice "no tengo información"
```

MultiQueryRetriever amplía el recall generando variaciones del query que cubren diferentes formulaciones.

---

## Cómo funciona

```
Query original: "¿Qué ventajas tiene LCEL?"
      │
      ▼
  LLM genera variaciones:
    1. "¿Cuáles son los beneficios de LangChain Expression Language?"
    2. "¿Por qué usar LCEL en lugar de chains legacy?"
    3. "¿Qué mejoras aporta LCEL al desarrollo?"
      │
      ▼
  Retrieval por cada variación (3 llamadas al retriever)
      │
      ▼
  Deduplicación (elimina chunks repetidos)
      │
      ▼
  Contexto ampliado para el LLM
```

---

## Implementación

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = get_llm()

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

# Se usa exactamente igual que cualquier retriever
docs = mq_retriever.invoke("¿Qué ventajas tiene LCEL?")
```

Internamente, `from_llm()` usa un prompt predefinido para pedirle al LLM que genere 3 alternativas del query. Puedes customizarlo pasando tu propio prompt con `prompt=`.

---

## Ver las queries generadas

```python
import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

docs = mq_retriever.invoke("mi pregunta")
# INFO: Generated queries: ['variación 1', 'variación 2', 'variación 3']
```

Activa esto durante el desarrollo para depurar qué variaciones está generando el LLM.

---

## Costo real

| Paso | Llamadas LLM | Llamadas retriever |
|------|-------------|-------------------|
| Generar variaciones | 1 | 0 |
| Retrieval por variación | 0 | N (3-5) |
| **Total** | **1 extra** | **3-5× más** |

El retriever local (ChromaDB) es barato. La llamada LLM extra para generar variaciones es el costo principal. Vale la pena cuando el recall es bajo o las queries son ambiguas.

---

## Cuándo usar Multi-Query

| Situación | Usar |
|-----------|------|
| Queries cortos y ambiguos | ✅ |
| Usuarios que formulan preguntas de forma imprecisa | ✅ |
| Documentos técnicos con terminología variada | ✅ |
| Queries muy específicos y precisos | ❌ Overhead sin ganancia |
| Latencia crítica | ❌ Agrega ~1 llamada LLM |
| Corpus pequeño (< 100 docs) | ❌ Poca ganancia en recall |

---

## Tradeoffs

| Ventaja | Contra |
|---------|--------|
| Mayor recall sin cambiar el corpus | +1 llamada LLM por query |
| Sin cambios en la ingesta | Latencia aumenta |
| Funciona sobre cualquier retriever | Las variaciones malas pueden recuperar docs irrelevantes |
| Transparente para el LLM | No mejora si el problema es el chunk size |
