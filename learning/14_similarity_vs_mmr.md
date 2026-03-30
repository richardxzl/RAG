# Similarity vs MMR — Estrategias de búsqueda en el vector store

## El problema con similarity puro

La búsqueda por similitud retorna los K chunks más cercanos al query en el espacio vectorial. Si el documento tiene 3 párrafos casi idénticos sobre el mismo tema, los 3 entran en el resultado — desperdiciando tokens de contexto con información redundante.

```
Query: "¿Qué es la fotosíntesis?"

Similarity (k=4):
  1. "La fotosíntesis es el proceso por el cual..."  ← relevante
  2. "El proceso de fotosíntesis consiste en..."     ← casi igual al 1
  3. "Durante la fotosíntesis, las plantas..."       ← casi igual al 1
  4. "La clorofila interviene en la fotosíntesis..." ← algo diferente
```

El LLM recibe 3 chunks que dicen lo mismo. Tokens desperdiciados, sin ganancia.

---

## MMR — Maximum Marginal Relevance

MMR balancea dos criterios al seleccionar cada chunk:
1. **Similitud con el query** (relevancia)
2. **Disimilitud con los chunks ya seleccionados** (diversidad)

Fórmula conceptual para cada candidato:
```
score = λ × similitud(chunk, query) - (1-λ) × max_similitud(chunk, ya_seleccionados)
```

- `λ = 1.0` → solo relevancia (equivale a similarity)
- `λ = 0.0` → solo diversidad
- `λ = 0.5` → balance (recomendado)

---

## Configuración en LangChain

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,          # chunks finales a retornar
        "fetch_k": 20,   # candidatos a evaluar antes de aplicar MMR (debe ser > k)
        "lambda_mult": 0.5,
    },
)
```

`fetch_k` es importante: MMR necesita un pool de candidatos para elegir. Si `fetch_k = k`, no tiene margen para diversificar. Un valor de `fetch_k = 4 × k` es un buen punto de partida.

---

## Cuándo usar cada estrategia

| Situación | Similarity | MMR |
|-----------|-----------|-----|
| Pregunta muy específica con respuesta puntual | ✅ | — |
| Pregunta amplia que requiere múltiples ángulos | — | ✅ |
| Documentos con mucha redundancia | — | ✅ |
| Resúmenes de documentos largos | — | ✅ |
| Prioridad absoluta: el chunk más relevante | ✅ | — |

---

## Tradeoffs

| | Similarity | MMR |
|---|---|---|
| **Velocidad** | Más rápida (un solo ranking) | Ligeramente más lenta (selección iterativa) |
| **Redundancia** | Alta si el doc tiene repetición | Baja |
| **Cobertura** | Puede perderse ángulos | Cubre más perspectivas |
| **Configuración** | Solo `k` | `k`, `fetch_k`, `lambda_mult` |

---

## Regla práctica

> Para RAG en producción, empieza con MMR `λ=0.5` y `fetch_k = 4×k`. Si ves que el LLM dice "no tengo información suficiente" en preguntas que deberían estar respondidas, sube `λ` hacia 1.0. Si ves respuestas repetitivas o el LLM "da vueltas", baja `λ`.
