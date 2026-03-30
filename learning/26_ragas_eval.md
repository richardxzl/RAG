# RAGAS — Evaluación Automatizada End-to-End

## Qué es RAGAS

RAGAS (Retrieval Augmented Generation Assessment) es una librería especializada para evaluar pipelines RAG. Implementa las métricas del módulo 5.2 de forma estandarizada y optimizada.

```bash
pip install ragas datasets
```

---

## Las métricas de RAGAS

| Métrica | Necesita ground_truth | Qué mide |
|---------|----------------------|----------|
| `faithfulness` | No | ¿La respuesta está soportada por el contexto? |
| `answer_relevancy` | No | ¿La respuesta responde la pregunta? |
| `context_precision` | No | ¿Los chunks recuperados son relevantes? |
| `context_recall` | **Sí** | ¿El contexto cubre el ground truth? |
| `answer_correctness` | **Sí** | ¿La respuesta coincide con el ground truth? |

Las métricas sin ground_truth son útiles cuando no tienes un dataset de evaluación completo.

---

## Estructura del dataset RAGAS

```python
dataset = [
    {
        "question": "¿Qué es LCEL?",
        "answer": "LCEL es el sistema de composición...",          # generado por tu RAG
        "contexts": ["LCEL usa el operador |...", "..."],         # chunks recuperados
        "ground_truth": "LCEL (LangChain Expression Language)...", # del dataset 5.1
    },
    # ...
]
```

---

## Flujo completo

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# 1. Preparar dataset ejecutando el RAG sobre las preguntas
dataset_ragas = []
for sample in eval_samples:
    docs = retriever.invoke(sample["question"])
    answer = chain.invoke({"question": sample["question"], "docs": docs})
    dataset_ragas.append({
        "question": sample["question"],
        "answer": answer,
        "contexts": [d.page_content for d in docs],
        "ground_truth": sample["ground_truth"],
    })

# 2. Evaluar
hf_dataset = Dataset.from_list(dataset_ragas)
resultado = evaluate(
    dataset=hf_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

# 3. Ver resultados
print(resultado)
df = resultado.to_pandas()
print(df[["question", "faithfulness", "answer_relevancy"]].head())
```

---

## Configurar el LLM evaluador

RAGAS usa OpenAI por defecto. Para usar Claude (langchain_anthropic):

```python
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic

llm_wrapper = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001"))

resultado = evaluate(
    dataset=hf_dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm_wrapper,
)
```

---

## RAGAS vs métricas manuales

| | RAGAS | Métricas manuales (módulo 5.2) |
|---|---|---|
| Setup | `pip install ragas` | Solo LangChain |
| Estándar | Sí — comparables entre proyectos | No — tu implementación |
| Customización | Limitada | Total |
| Documentación | Amplia | La que escribas tú |
| Cuándo usar | Evaluación rápida estándar | Entender cómo funcionan o necesitar métricas custom |

---

## Generación sintética de dataset

RAGAS puede generar el dataset de evaluación automáticamente desde tus documentos:

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.with_langchain(
    generator_llm=llm,
    critic_llm=llm,
    embeddings=embeddings,
)

testset = generator.generate_with_langchain_docs(
    docs,
    test_size=20,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)
```

Genera preguntas de tres tipos: simples, que requieren razonamiento, y que requieren múltiples chunks.
