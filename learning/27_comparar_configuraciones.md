# Comparar Configuraciones — Grid Search sobre el pipeline RAG

## Por qué hacer un grid search

El pipeline RAG tiene múltiples hiperparámetros que interactúan entre sí. Cambiar `chunk_size` afecta qué tan buenas son las búsquedas. Cambiar `k` afecta cuánto contexto recibe el LLM. No puedes optimizarlos independientemente.

Un grid search evalúa todas las combinaciones y te dice cuál da mejores resultados **en tu corpus y con tus queries**.

---

## Espacio de configuraciones

```python
CHUNK_SIZES      = [500, 1000, 1500]
RETRIEVER_TYPES  = ["similarity", "mmr"]
K_VALUES         = [3, 5, 8]

# 3 × 2 × 3 = 18 configuraciones
# Con 10 preguntas de evaluación = 180 ejecuciones RAG + 180 evaluaciones LLM
```

Esto tiene un costo real en tiempo y tokens. Para proyectos de aprendizaje, usa un subset:

```python
CHUNK_SIZES = [500, 1000]    # 2
RETRIEVER_TYPES = ["similarity", "mmr"]  # 2
K_VALUES = [3, 5]            # 2
# = 8 combinaciones — manejable
```

---

## Flujo del grid search

```
Para cada (chunk_size, retriever_type, k):
  1. Re-chunkear los docs con ese chunk_size
  2. Indexar en vector store temporal
  3. Para cada pregunta del dataset:
     a. Recuperar k docs
     b. Generar respuesta con el LLM
     c. Evaluar faithfulness y relevance (LLM-as-a-judge)
  4. Promediar métricas
  5. Rankear configuraciones
```

---

## Interpretar los resultados

| Observación | Causa probable | Acción |
|-------------|---------------|--------|
| Faithfulness baja en todas | El LLM alucina — problema de prompt | Mejorar el prompt RAG |
| Faithfulness buena, relevance baja | El retriever no encuentra los docs correctos | Cambiar retriever o chunk_size |
| MMR consistentemente peor que similarity | Las queries son precisas — no necesitas diversidad | Usar similarity |
| k grande no mejora (o empeora) | Los docs extra agregan ruido | Reducir k o usar compresión (módulo 3.5) |
| chunk_size pequeño gana | Las respuestas son puntuales | Reducir chunk_size |
| chunk_size grande gana | Las respuestas requieren contexto amplio | Aumentar chunk_size |

---

## Costo del grid search

Con `n_configs × n_preguntas` ejecuciones, el costo crece rápido:

```
8 configs × 10 preguntas = 80 llamadas RAG + 80 evaluaciones = 160 llamadas LLM
```

Estrategias para reducir costo:
- Usa un modelo económico para el evaluador (claude-haiku)
- Empieza con 3-5 preguntas representativas, no el dataset completo
- Fija un parámetro a la vez (no todas las combinaciones juntas)

---

## Conexión con regression testing (módulo 5.5)

El grid search te dice cuál es la mejor configuración hoy. El regression testing te avisa si esa configuración empeora después de un cambio en el pipeline.

**Workflow recomendado**:
1. Grid search → encontrar config óptima
2. Fijar esa config como baseline
3. Regression testing → detectar regresiones en futuros cambios
