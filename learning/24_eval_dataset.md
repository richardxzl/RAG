# Dataset de Evaluación — Ground Truth para RAG

## Por qué necesitas un dataset

Sin ground truth no puedes medir nada. Puedes tener una intuición de que el RAG "funciona bien", pero sin números no sabes si mejoró o empeoró después de un cambio.

El dataset de evaluación es la base de verdad contra la que comparas las respuestas del sistema.

---

## Estructura de un EvalSample

```json
{
  "id": "q001",
  "question": "¿Qué es LCEL?",
  "ground_truth": "LCEL (LangChain Expression Language) es el sistema de composición moderno...",
  "metadata": {
    "tipo": "conceptual",
    "dificultad": "baja",
    "modulo": "1"
  }
}
```

| Campo | Obligatorio | Para qué sirve |
|-------|------------|----------------|
| `id` | ✅ | Identificar el sample en reportes |
| `question` | ✅ | La pregunta que el RAG debe responder |
| `ground_truth` | ✅ | La respuesta correcta de referencia |
| `metadata.tipo` | Recomendado | Segmentar métricas por tipo de pregunta |
| `metadata.dificultad` | Recomendado | Detectar si el sistema falla en preguntas difíciles |

---

## Tipos de preguntas a incluir

| Tipo | Ejemplo | Por qué incluirlo |
|------|---------|------------------|
| Factual / conceptual | "¿Qué es LCEL?" | Prueba el retrieval básico |
| Comparativo | "¿Diferencia entre X e Y?" | Requiere múltiples chunks |
| Aplicación | "¿Cuándo usar X?" | Requiere razonamiento sobre el contexto |
| Negativo | "¿Qué NO puede hacer X?" | Detecta alucinaciones |
| Fuera de scope | "¿Quién ganó el Mundial?" | Verifica que el sistema reconoce los límites |

---

## Cómo generar el ground truth

**Opción A — Manual (más confiable)**: un experto del dominio escribe las respuestas. Lento pero preciso.

**Opción B — LLM + revisión humana**: el LLM genera respuestas sobre los documentos originales, un humano las revisa y corrige. Más rápido.

**Opción C — RAGAS synthetic generation**: RAGAS puede generar preguntas + respuestas automáticamente desde los documentos. Útil para corpus grandes, pero requiere revisión.

```python
# Generación sintética con RAGAS (módulo 5.3)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning

generator = TestsetGenerator.with_openai()
testset = generator.generate_with_langchain_docs(docs, test_size=20)
```

---

## Cuántos samples necesitas

| Dataset | Uso |
|---------|-----|
| 10-20 samples | Smoke test durante desarrollo |
| 50-100 samples | Evaluación de una versión específica |
| 200+ samples | Comparación estadísticamente significativa |

Para la mayoría de proyectos de aprendizaje o MVPs, 20-50 samples bien diseñados son suficientes.

---

## Formato de persistencia

El dataset se guarda como JSON para ser portable y legible:

```python
dataset = {
    "version": "1.0",
    "created_at": "2026-03-30T...",
    "total": 8,
    "samples": [...]
}

Path("eval_dataset.json").write_text(json.dumps(dataset, indent=2))
```

---

## Regla de oro

> Si no puedes escribir el ground truth para una pregunta, es una señal de que el dominio no está bien definido o que los documentos no contienen esa información. Esas preguntas no deberían estar en el RAG — primero define el scope, luego evalúa.
