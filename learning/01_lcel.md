# LCEL — LangChain Expression Language

## ¿Qué es?

LCEL es el sistema de composición moderno de LangChain. Reemplaza las "chains" legacy (`RetrievalQA`, `LLMChain`, etc.) por un patrón declarativo basado en el operador pipe (`|`).

Cada pieza del pipeline es un **Runnable** — un bloque con `.invoke()`, `.stream()`, `.batch()` — que se conecta con otros.

## ¿Por qué se creó?

LangChain legacy tenía un problema: cada tipo de chain era una clase con su propia API, parámetros y comportamiento. Si querías modificar el flujo, tenías que heredar, sobrescribir métodos, o buscar parámetros ocultos.

LCEL unifica todo bajo una sola interfaz: **Runnable**. Cualquier cosa que tenga `.invoke()` se puede componer.

## Antes vs Después

### Legacy (RetrievalQA)

```python
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": RAG_PROMPT},
)

result = chain.invoke({"query": "mi pregunta"})
# result["result"] → respuesta
# result["source_documents"] → docs
```

**Problemas:**
- `chain_type="stuff"` — ¿Qué significa? Es un nombre opaco para "meter todos los docs en el prompt".
- `chain_type_kwargs` — Parámetros pasados como dict genérico.
- No puedes insertar pasos intermedios fácilmente (ej: logging, transformaciones).
- Streaming requiere configuración especial.

### LCEL (moderno)

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("mi pregunta")
# answer → string directo
```

**Ventajas:**
- Lees el flujo de arriba a abajo: recuperar → formatear → prompt → LLM → parsear.
- Insertar un paso es agregar una línea con `|`.
- Streaming gratis: `rag_chain.stream("mi pregunta")`.
- Batch gratis: `rag_chain.batch(["pregunta1", "pregunta2"])`.

## Componentes clave de LCEL

### RunnablePassthrough

Pasa el input tal cual al siguiente paso. Útil cuando necesitas que un valor "pase de largo" mientras otros se transforman.

```python
# La pregunta pasa sin modificar, el contexto se busca y formatea
RunnableParallel(
    context=retriever | format_docs,   # se transforma
    question=RunnablePassthrough(),     # pasa tal cual
)
```

### RunnableParallel

Ejecuta múltiples runnables en paralelo y combina sus resultados en un dict.

```python
# Estos dos se ejecutan AL MISMO TIEMPO:
RunnableParallel(
    context=retriever | format_docs,
    question=RunnablePassthrough(),
)
# Output: {"context": "texto de docs...", "question": "la pregunta"}
```

### RunnablePassthrough.assign()

Toma el input (un dict) y le agrega nuevas keys sin perder las existentes.

```python
# Input: {"question": "...", "chat_history": [...]}
# Output: {"question": "...", "chat_history": [...], "context": "docs formateados"}
RunnablePassthrough.assign(
    context=lambda x: format_docs(retriever.invoke(x["question"])),
)
```

### RunnableLambda

Envuelve cualquier función Python en un Runnable.

```python
from langchain_core.runnables import RunnableLambda

def log_input(x):
    print(f"Input recibido: {x}")
    return x

chain = RunnableLambda(log_input) | prompt | llm
```

### StrOutputParser

Extrae el string de la respuesta del LLM (que normalmente viene como un objeto `AIMessage`).

```python
# Sin parser: AIMessage(content="la respuesta", ...)
# Con parser: "la respuesta"
chain = prompt | llm | StrOutputParser()
```

## Beneficios

| Beneficio | Detalle |
|-----------|---------|
| **Composabilidad** | Cualquier Runnable se conecta con `\|` |
| **Streaming nativo** | `.stream()` funciona en toda la cadena sin config extra |
| **Batch nativo** | `.batch()` para procesar múltiples inputs en paralelo |
| **Tipado** | Cada Runnable tiene input/output types definidos |
| **Debugging** | Puedes inspeccionar cada paso individualmente con `.invoke()` |
| **Async** | `.ainvoke()`, `.astream()` para aplicaciones async |

## Contras / Tradeoffs

| Contra | Detalle |
|--------|---------|
| **Curva de aprendizaje** | El operador `\|` y los Runnables son un paradigma nuevo |
| **Verbose para cosas simples** | Un RAG básico es más líneas que `RetrievalQA.from_chain_type()` |
| **Debugging implícito** | Si algo falla en medio del pipe, el traceback puede ser confuso |
| **Acoplamiento a LangChain** | Todo debe ser un Runnable — integrar código externo requiere `RunnableLambda` |

## ¿Cuándo usar qué?

- **Prototipo rápido, flujo simple** → Legacy chains todavía funcionan (pero están deprecated).
- **Producción, flujos personalizados, streaming** → LCEL es el camino.
- **Flujos con decisiones, loops, estado** → LangGraph (construido sobre LCEL).
