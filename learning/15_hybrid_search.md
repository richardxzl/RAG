# Hybrid Search — BM25 + Embeddings

## El problema de cada uno por separado

**Solo embeddings**: fallan con términos técnicos exactos. Si el usuario pregunta por "UUID v4" o "RFC 7519", el embedding puede no capturar la especificidad del término y recuperar chunks sobre identificadores en general.

**Solo BM25**: falla con paráfrasis. "¿Cómo se autentica el usuario?" y "proceso de login" son la misma pregunta, pero BM25 no lo ve porque son palabras distintas.

La solución es combinarlos.

---

## BM25 — Búsqueda por palabras clave

BM25 (Best Match 25) es la evolución moderna de TF-IDF. Rankea documentos según la frecuencia de los términos del query en el documento, con normalización por longitud.

```python
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs, k=4)
resultados = bm25.invoke("LCEL pipeline")
```

**Características**:
- Opera sobre texto, no vectores — no necesita embeddings
- Exacto: busca esas palabras, no sinónimos
- Rápido en CPU, sin GPU
- Requiere los documentos en memoria (no es un vector store persistente)

---

## EnsembleRetriever — Fusión con RRF

`EnsembleRetriever` combina múltiples retrievers usando **Reciprocal Rank Fusion (RRF)**:

```
score_rrf(doc) = Σ_retriever  1 / (k + rank_del_doc_en_ese_retriever)
```

Los documentos que aparecen en el top de varios retrievers a la vez reciben un score mayor.

```python
from langchain.retrievers import EnsembleRetriever

hybrid = EnsembleRetriever(
    retrievers=[bm25, embedding_retriever],
    weights=[0.4, 0.6],  # embeddings tiene más peso
)
```

Los `weights` escalan los scores RRF antes de fusionar. No son probabilidades — son multiplicadores de relevancia.

---

## Flujo completo

```python
# 1. Cargar docs para BM25 (necesita el texto en memoria)
docs = vectorstore.get(include=["documents", "metadatas"])

# 2. Crear BM25
bm25 = BM25Retriever.from_documents(docs, k=4)

# 3. Crear embedding retriever (ChromaDB)
embedding = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Combinar
hybrid = EnsembleRetriever(
    retrievers=[bm25, embedding],
    weights=[0.4, 0.6],
)

# 5. Invocar — igual que cualquier retriever
docs = hybrid.invoke("mi query")
```

---

## Cuándo usar cada estrategia

| Tipo de query | BM25 | Embeddings | Hybrid |
|---------------|------|-----------|--------|
| Términos técnicos exactos | ✅ | ⚠️ | ✅ |
| Paráfrasis / sinónimos | ❌ | ✅ | ✅ |
| Nombres propios / siglas | ✅ | ⚠️ | ✅ |
| Preguntas conceptuales largas | ❌ | ✅ | ✅ |
| Queries en idioma diferente al doc | ❌ | ✅ (cross-lingual) | ⚠️ |

**Regla general**: si no sabes qué tipo de queries recibirás, usa hybrid. El costo es mínimo y raramente empeora.

---

## Tradeoffs

| Ventaja | Detalle |
|---------|---------|
| Mejor recall | Cubre los puntos ciegos de cada técnica |
| Robusto a variaciones | Términos exactos + paráfrasis |

| Contra | Detalle |
|--------|---------|
| BM25 en memoria | Los documentos deben caber en RAM |
| Más lento | Dos retrievals + fusión |
| Tuning de pesos | `weights` requiere experimentación por dominio |
| Sin persistencia para BM25 | Hay que reconstruirlo en cada arranque (o usar índices como Elasticsearch) |
