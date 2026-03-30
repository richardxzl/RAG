# Nodo Generator — Genera respuesta con contexto

## Responsabilidades

1. Leer `documentos_relevantes` (o `documentos` como fallback) del estado
2. Formatear el contexto
3. Invocar el LLM con el prompt RAG
4. Escribir `respuesta` y `contexto` en el estado

---

## Generator básico

```python
PROMPT = ChatPromptTemplate.from_template(
    """Responde SOLO basándote en el contexto. Si no está, di que no tienes información.
    Contexto: {context}
    Pregunta: {question}
    Respuesta:"""
)

def nodo_generator(estado):
    contexto = format_docs(estado["documentos_relevantes"] or estado["documentos"])
    respuesta = (PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content
    return {"contexto": contexto, "respuesta": respuesta}
```

---

## Generator con fuentes citadas

```python
def nodo_generator_con_fuentes(estado):
    fuentes = list({doc.metadata.get("source", "?") for doc in estado["documentos"]})
    respuesta = (PROMPT_CON_FUENTES | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
        "fuentes": ", ".join(fuentes),
    }).content
    return {"respuesta": respuesta, "fuentes_citadas": fuentes}
```

---

## Generator con evaluación de confianza

```python
PROMPT_CONFIANZA = ChatPromptTemplate.from_template(
    """Responde y evalúa tu confianza.
    Contexto: {context}
    Pregunta: {question}

    RESPUESTA: [tu respuesta]
    CONFIANZA: alta|media|baja — [razón]"""
)

def nodo_generator_con_confianza(estado):
    raw = (PROMPT_CONFIANZA | llm).invoke(...).content
    # Parsear RESPUESTA: y CONFIANZA: del output
    respuesta = extraer_linea(raw, "RESPUESTA:")
    confianza = "alta" if "alta" in raw else "baja" if "baja" in raw else "media"
    return {"respuesta": respuesta, "confianza": confianza}
```

---

## ¿Cuál usar?

| Modo | Cuándo |
|------|--------|
| Básico | Prototipo, uso interno |
| Con fuentes | Usuarios que necesitan verificar |
| Con confianza | Sistemas donde la incertidumbre importa (médico, legal) |

---

## Pregunta original vs pregunta reformulada

Cuando el sistema reformula la pregunta para mejorar el retrieval, el Generator debe responder a la **pregunta original**, no a la reformulada:

```python
def nodo_generator(estado):
    # Siempre responde a la pregunta que hizo el usuario
    respuesta = (PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["pregunta_original"],  # no pregunta_actual
    }).content
```
