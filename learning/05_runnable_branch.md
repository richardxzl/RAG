# RunnableBranch — Condicionales en el pipeline LCEL

## El problema que resuelve

Un pipeline RAG lineal aplica siempre los mismos pasos: retriever → prompt → LLM. Pero no todas las preguntas necesitan RAG:

- **"Hola, ¿cómo estás?"** — No tiene sentido buscar documentos para un saludo.
- **"¿Cuál es el pronóstico del tiempo?"** — No tiene sentido buscar documentos de un dominio que no manejas.
- **"¿Qué dice el documento sobre autenticación?"** — Aquí sí necesitas el retriever.

Sin `RunnableBranch`, tendrías que partir el pipeline con `if/else` fuera de LCEL, perdiendo composabilidad:

```python
# Sin RunnableBranch — lógica partida, no composable
def query(question: str) -> str:
    if es_saludo(question):
        return llm.invoke(CONV_PROMPT.format(question=question))
    elif fuera_de_scope(question):
        return llm.invoke(SCOPE_PROMPT.format(question=question))
    else:
        docs = retriever.invoke(question)
        return llm.invoke(RAG_PROMPT.format(context=format_docs(docs), question=question))
```

`RunnableBranch` mete esa lógica condicional DENTRO del pipeline, manteniendo todo composable, observble y compatible con `.batch()`, `.stream()` y `.ainvoke()`.

---

## Sintaxis

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (condicion_1, runnable_1),
    (condicion_2, runnable_2),
    (condicion_3, runnable_3),
    runnable_default,   # sin tupla — es el fallback obligatorio
)
```

Cada entrada es una tupla `(condicion, runnable)`, excepto el último elemento que es el **runnable por defecto** (sin condición). Es obligatorio incluirlo.

La condición puede ser:
- Una función `callable` que recibe el input y retorna `bool`
- Un `Runnable` cuyo output se evalúa como truthy

---

## Cómo se evalúan las condiciones

**En orden, la primera que retorne `True` gana.** Las demás no se evalúan.

```
input
  │
  ├─ condicion_1(input) == True?  →  runnable_1.invoke(input) → output
  │
  ├─ condicion_2(input) == True?  →  runnable_2.invoke(input) → output
  │
  ├─ condicion_3(input) == True?  →  runnable_3.invoke(input) → output
  │
  └─ (ninguna) →  runnable_default.invoke(input) → output
```

El input que recibe el runnable seleccionado es el **mismo input original** que recibió el `RunnableBranch`. No es el resultado de la condición — es el input completo.

---

## Diagrama ASCII del flujo (caso de este proyecto)

```
pregunta del usuario
        │
        ▼
  {"question": "..."}
        │
        ▼
┌───────────────────────────────────────────┐
│            RunnableBranch                 │
│                                           │
│  ¿keywords fuera de scope?                │
│       │                                   │
│       ├── SÍ ──► FUERA_SCOPE_PROMPT       │
│       │              │                    │
│       │              ▼                    │
│       │             LLM ──► StrOutput     │
│       │                                   │
│  ¿keywords conversacional?                │
│       │                                   │
│       ├── SÍ ──► CONV_PROMPT              │
│       │              │                    │
│       │              ▼                    │
│       │             LLM ──► StrOutput     │
│       │                                   │
│  DEFAULT (factual)                        │
│       │                                   │
│       └──────► retriever                  │
│                    │                      │
│                    ▼                      │
│               format_docs                 │
│                    │                      │
│                    ▼                      │
│              QUERY_PROMPT                 │
│                    │                      │
│                    ▼                      │
│                   LLM ──► StrOutput       │
└───────────────────────────────────────────┘
        │
        ▼
   respuesta final (str)
```

---

## Condición que retorna bool vs condición que retorna el valor

Hay dos formas válidas de escribir la condición:

### Forma 1: función que retorna bool (más legible)

```python
def es_conversacional(input_data: dict) -> bool:
    question = input_data.get("question", "").lower()
    return any(kw in question for kw in ["hola", "gracias", "bye"])

branch = RunnableBranch(
    (es_conversacional, rama_conversacional),
    rama_factual,
)
```

### Forma 2: Runnable cuyo output es truthy

```python
# Un RunnableLambda también puede ser la condición
from langchain_core.runnables import RunnableLambda

detectar = RunnableLambda(lambda x: "hola" in x["question"].lower())

branch = RunnableBranch(
    (detectar, rama_conversacional),
    rama_factual,
)
```

**Recomendación**: usa funciones que retornan `bool` (Forma 1). Es más explícito, más fácil de testear unitariamente, y el intent queda claro.

---

## Casos de uso reales

### 1. Router de intenciones (este proyecto)

Clasifica la pregunta antes de decidir si usar RAG o responder directamente:

```python
branch = RunnableBranch(
    (lambda x: es_fuera_de_scope(x), rama_sin_rag_scope),
    (lambda x: es_conversacional(x), rama_sin_rag_conv),
    rama_rag_completo,  # default
)
```

### 2. A/B de prompts según idioma

```python
branch = RunnableBranch(
    (lambda x: detectar_idioma(x["question"]) == "en", prompt_en | llm | parser),
    (lambda x: detectar_idioma(x["question"]) == "fr", prompt_fr | llm | parser),
    prompt_es | llm | parser,  # default: español
)
```

### 3. Fallback por tipo de input

```python
branch = RunnableBranch(
    (lambda x: len(x["question"]) > 500, resumir_primero | rag_chain),
    (lambda x: x.get("es_codigo", False), code_chain),
    rag_chain,
)
```

### 4. Nivel de detalle según perfil de usuario

```python
branch = RunnableBranch(
    (lambda x: x["perfil"] == "experto",   prompt_tecnico | llm | parser),
    (lambda x: x["perfil"] == "principiante", prompt_basico | llm | parser),
    prompt_intermedio | llm | parser,
)
```

---

## Implementación en este proyecto

Ver [05_runnable_branch.py](../../05_runnable_branch.py) para la demo completa.

El clasificador de intención es puramente sintáctico (sin LLM):

```python
def detectar_intencion(input_data: dict) -> str:
    question = input_data.get("question", "").lower()
    palabras = set(question.split())

    if palabras & FUERA_DE_SCOPE_KEYWORDS:
        return "fuera_de_scope"
    if palabras & CONVERSACIONAL_KEYWORDS:
        return "conversacional"
    if len(palabras) < 4:
        return "conversacional"
    return "factual"
```

¿Por qué no llamar al LLM para clasificar? Porque agregar una llamada al LLM solo para clasificar suma ~500ms de latencia antes de la respuesta real. Para intenciones simples, las reglas keyword-based son más rápidas, deterministas y sin costo adicional de tokens. Para sistemas de producción más complejos, considera modelos de clasificación locales livianos (fastText, SetFit) antes de escalar a un LLM clasificador.

---

## Limitaciones

### 1. Evaluación lineal, no es un árbol

`RunnableBranch` evalúa condiciones **en secuencia**. No es un árbol de decisiones:

```
# Esto NO es posible con RunnableBranch:
#
#        ¿técnico?
#        /       \
#    ¿código?  ¿concepto?
#    /    \       |
# rama1  rama2  rama3
```

Para lógica de routing anidada o multi-nivel, usa **LangGraph** que modela el flujo como un grafo dirigido real.

### 2. Las condiciones no pueden compartir estado entre sí

Cada condición recibe el input original. No puedes hacer:

```python
# Esto NO funciona: la primera condición no puede "pasar datos" a la segunda
RunnableBranch(
    (lambda x: clasificar_y_guardar(x),  rama_1),  # guarda en algún lado
    (lambda x: leer_lo_guardado(x),      rama_2),  # intenta leerlo — MAL
    rama_default,
)
```

Si necesitas que la clasificación genere datos que consuma el runnable, usa `RunnablePassthrough.assign` antes del branch:

```python
pipeline = (
    RunnablePassthrough.assign(intencion=RunnableLambda(detectar_intencion))
    | RunnableBranch(
        (lambda x: x["intencion"] == "factual", rama_factual),
        rama_default,
    )
)
```

### 3. No soporta condiciones async nativamente

Las funciones de condición deben ser síncronas. Si necesitas condiciones async (consultar una base de datos para decidir), estructura el pipeline diferente o usa LangGraph.

---

## Tradeoffs

| Ventaja | Detalle |
|---------|---------|
| Condicionales dentro del pipeline | La lógica de routing queda encapsulada en el Runnable, no en código exterior |
| Composable | Se puede insertar en cualquier punto del pipe con `\|` |
| Compatible con `.batch()` / `.stream()` / `.ainvoke()` | Hereda toda la interfaz Runnable |
| Testeable por rama | Cada runnable de cada rama se puede testear de forma independiente |

| Limitación | Detalle |
|------------|---------|
| Solo lineal | No es un árbol; para routing complejo usa LangGraph |
| Condiciones evaluadas N veces | Si la clasificación es costosa, cada condición la llama de nuevo; extrae la clasificación con `.assign()` antes |
| Sin estado compartido entre condiciones | Las condiciones no se comunican entre sí |
| Condiciones síncronas | No soporta `async` en las funciones de condición |

---

## Cuándo usar RunnableBranch vs LangGraph

| Escenario | Usa |
|-----------|-----|
| 2-4 ramas simples, condiciones baratas | `RunnableBranch` |
| Routing anidado o multi-nivel | LangGraph |
| Ciclos o pasos que dependen del output de otros | LangGraph |
| Necesitas observabilidad granular por nodo | LangGraph |
| Lógica de retry o fallback con estado | LangGraph |

> **Regla de oro**: si puedes dibujar el flujo como una lista de `if/elif/else`, usa `RunnableBranch`. Si necesitas un grafo real con nodos y aristas, usa LangGraph.
