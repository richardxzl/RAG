# Few-shot Prompting — Enseñar con ejemplos

## Por qué funciona

El LLM no "aprende" durante la inferencia — sus pesos son fijos. Lo que hacen los ejemplos es **acotar el espacio de respuestas posibles**: el modelo ve el patrón `input → output` repetido N veces y lo replica para el nuevo input.

Es una forma de programar el comportamiento sin modificar el modelo.

```
Sin ejemplos (zero-shot):
  "Clasifica: 'Producto excelente'" → "El sentimiento es positivo, ya que..."

Con ejemplos (few-shot):
  [Ejemplo 1] "Llegó roto" → "negativo"
  [Ejemplo 2] "Superó expectativas" → "positivo"
  "Clasifica: 'Producto excelente'" → "positivo"
```

El few-shot fuerza el formato de salida implícitamente. No necesitas decirle "responde solo con una palabra" si todos los ejemplos tienen solo una palabra.

---

## FewShotPromptTemplate vs FewShotChatMessagePromptTemplate

| | FewShotPromptTemplate | FewShotChatMessagePromptTemplate |
|---|---|---|
| Para | Modelos de completion (legacy) | Chat models modernos |
| Output | String formateado | Lista de mensajes alternados human/ai |
| Rol de los ejemplos | Texto plano en el prompt | Pares HumanMessage / AIMessage |

Para Claude, GPT-4, o cualquier modelo chat, usa siempre `FewShotChatMessagePromptTemplate`.

---

## Estructura de un ejemplo

Un ejemplo es un `dict` con las mismas keys que el `example_prompt` espera:

```python
ejemplo_template = ChatPromptTemplate.from_messages([
    ("human", "{resena}"),   # ← key: "resena"
    ("ai", "{sentimiento}"), # ← key: "sentimiento"
])

ejemplos = [
    {"resena": "Llegó roto.", "sentimiento": "negativo"},
    {"resena": "Excelente producto.", "sentimiento": "positivo"},
]
```

La consistencia entre los ejemplos es CRÍTICA. Si algunos tienen "Positivo" y otros "positivo", el LLM puede seguir cualquiera de los dos patrones.

---

## Few-shot estático

Los ejemplos son una lista fija que siempre se incluye completa:

```python
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=ejemplo_template,
    examples=ejemplos,  # siempre estos, siempre todos
)

template_final = ChatPromptTemplate.from_messages([
    ("system", "Clasifica el sentimiento..."),
    few_shot,          # ← se expanden aquí como pares human/ai
    ("human", "{resena}"),
])
```

**Cuándo usar**: pocos ejemplos (3-6), ya curados, representativos de todos los casos.

---

## Few-shot dinámico con SemanticSimilarityExampleSelector

Cuando tienes muchos ejemplos, incluirlos todos aumenta el costo por token. El selector elige los K más relevantes para cada pregunta:

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

selector = SemanticSimilarityExampleSelector.from_examples(
    examples=ejemplos,          # pool completo
    embeddings=get_embeddings(),
    vectorstore_cls=Chroma,
    k=3,                        # usar los 3 más similares
)

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=ejemplo_template,
    example_selector=selector,  # en vez de examples=
)
```

El selector vectoriza todos los ejemplos una vez. En cada invocación, vectoriza la pregunta y hace búsqueda de similitud.

### MaxMarginalRelevanceExampleSelector

Alternativa que balancea relevancia y diversidad (evita seleccionar 3 ejemplos casi idénticos):

```python
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
# misma API, diferente algoritmo de selección
```

---

## Cuántos ejemplos usar

| Ejemplos | Efecto |
|----------|--------|
| 0 (zero-shot) | Máxima variabilidad en formato, menor consistencia |
| 1-2 (one/two-shot) | Ayuda con el formato, poco contexto de variación |
| 3-5 | Sweet spot: suficiente patrón, costo razonable |
| 10+ | Riesgo de "lost in the middle" — el LLM puede ignorar los centrales |

La calidad supera a la cantidad. 3 ejemplos bien elegidos > 10 ejemplos mediocres.

---

## Cuándo few-shot supera al fine-tuning

| Situación | Few-shot | Fine-tuning |
|-----------|----------|-------------|
| Tarea nueva, sin datos | ✅ | ❌ |
| Cambio de comportamiento rápido | ✅ | ❌ (requiere training) |
| Formato de output específico | ✅ | ✅ |
| Tarea muy específica con miles de ejemplos | ❌ (contexto limitado) | ✅ |
| Costo en producción (alto volumen) | ❌ (tokens por request) | ✅ (prompt más corto) |

---

## Tradeoffs

| Ventaja | Detalle |
|---------|---------|
| Sin entrenamiento | Funciona inmediatamente |
| Adaptable | Cambiar ejemplos cambia el comportamiento |
| Debuggeable | Puedes leer los ejemplos y entender por qué falla |

| Contra | Detalle |
|--------|---------|
| Costo por token | Cada request incluye todos los ejemplos |
| Ejemplos malos | Un ejemplo incorrecto contamina el patrón |
| Límite de contexto | No puedes escalar indefinidamente los ejemplos |
