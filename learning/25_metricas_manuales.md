# Métricas Manuales de Evaluación RAG

## Las 4 métricas fundamentales

Un sistema RAG tiene dos partes evaluables: el **retrieval** (¿encontró los docs correctos?) y la **generación** (¿produjo una buena respuesta?).

```
Query → [Retriever] → Chunks → [LLM] → Respuesta
           ↑                              ↑
    Context Precision              Faithfulness
    Context Recall                 Answer Relevance
```

| Métrica | Qué mide | Rango |
|---------|----------|-------|
| **Faithfulness** | ¿La respuesta está soportada por el contexto? | 0-1 |
| **Answer Relevance** | ¿La respuesta responde a la pregunta? | 0-1 |
| **Context Precision** | ¿Los chunks recuperados son relevantes? | 0-1 |
| **Context Recall** | ¿El ground truth está en los chunks? | 0-1 |

---

## Faithfulness — La métrica más importante

Mide si el LLM inventó algo o si todo lo que dijo está en el contexto. Una respuesta puede ser relevante pero no fiel — el LLM añadió información que no estaba en los chunks.

```
Context: "LCEL usa el operador |"
Answer: "LCEL usa el operador | y fue creado en 2023 por Adam D'Angelo"
                                   ↑ inventado — faithfulness baja
```

**Implementación con LLM-as-a-judge**:
```python
prompt = "¿Cada afirmación de la RESPUESTA está soportada por el CONTEXTO? Score 0-1."
score = llm.invoke({"context": context, "answer": answer})
```

---

## LLM-as-a-judge — Evaluar con el propio LLM

En lugar de comparación exacta de strings, usas el LLM para evaluar la calidad semántica. El LLM actúa como un evaluador humano.

```python
from langchain_core.output_parsers import PydanticOutputParser

class FaithfulnessScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    razon: str

parser = PydanticOutputParser(pydantic_object=FaithfulnessScore)
chain = eval_prompt | llm | parser

result = chain.invoke({"context": ctx, "answer": ans,
                       "format_instructions": parser.get_format_instructions()})
print(result.score)   # 0.85
print(result.razon)   # "La respuesta menciona X que sí está en el contexto..."
```

**Ventaja**: semántico — entiende paráfrasis y sinónimos.
**Contra**: el LLM evaluador puede equivocarse. Usa un modelo capaz (Claude Sonnet o GPT-4).

---

## Context Precision vs Context Recall

```
Ground truth menciona: [A, B, C]
Chunks recuperados contienen: [A, B, D, E]

Context Precision = docs relevantes / total docs recuperados = 2/4 = 0.5
Context Recall    = ground truth cubierto / total ground truth = 2/3 = 0.67
```

- **Precision alta, recall bajo**: el retriever es conservador — recupera pocos docs pero buenos
- **Recall alto, precision baja**: el retriever es agresivo — recupera muchos docs, muchos innecesarios

---

## Interpretación de scores

| Score | Interpretación |
|-------|---------------|
| 0.8 - 1.0 | Bueno — producción |
| 0.6 - 0.8 | Aceptable — revisar casos de fallo |
| 0.4 - 0.6 | Mejorable — hay un problema sistémico |
| < 0.4 | Malo — revisar pipeline completo |

---

## Cuándo usar métricas manuales vs RAGAS

| | Métricas manuales | RAGAS |
|---|---|---|
| Setup | Mínimo | `pip install ragas` |
| Control | Total | Limitado |
| Velocidad | Más lento (LLM por métrica) | Más rápido (optimizado) |
| Personalización | Alta | Media |
| Cuándo usar | Entender cómo funciona, customizar | Evaluación rápida en producción |
