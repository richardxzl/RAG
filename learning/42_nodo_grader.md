# Nodo Grader — ¿Los chunks son relevantes?

## El problema

El retriever devuelve los top-k documentos por similitud vectorial. Pero similitud no implica relevancia: puede devolver chunks que hablan del mismo tema general pero no responden la pregunta específica.

Sin Grader: generas con ruido → respuesta de baja calidad.
Con Grader: filtras el ruido → solo los chunks útiles llegan al Generator.

---

## Estrategia 1: Score de similitud

```python
SCORE_THRESHOLD = 0.7  # distancia ChromaDB (menor = más similar)

def nodo_grader_score(estado):
    relevantes = [
        doc for doc in estado["documentos"]
        if doc.metadata.get("retrieval_score", 1.0) <= SCORE_THRESHOLD
    ]
    return {
        "documentos_relevantes": relevantes,
        "todos_relevantes": len(relevantes) == len(estado["documentos"]),
    }
```

**Requiere**: que el nodo Retriever haya guardado el score en `doc.metadata["retrieval_score"]`.

---

## Estrategia 2: LLM judge

```python
GRADER_PROMPT = ChatPromptTemplate.from_template(
    """¿Este fragmento ayuda a responder la pregunta?
    Pregunta: {pregunta}
    Fragmento: {fragmento}
    Responde SOLO: relevante o irrelevante"""
)

def nodo_grader_llm(estado):
    relevantes = []
    for doc in estado["documentos"]:
        veredicto = (GRADER_PROMPT | llm).invoke({
            "pregunta": estado["pregunta"],
            "fragmento": doc.page_content[:500],
        }).content.strip().lower()

        if "relevante" in veredicto and "irrelevante" not in veredicto:
            relevantes.append(doc)

    return {"documentos_relevantes": relevantes}
```

**Costo**: una llamada al LLM por documento. Con k=4, son 4 llamadas extra.

---

## Edge condicional post-grader

```python
def decidir_post_grader(estado) -> Literal["generator", "reformular"]:
    if estado["documentos_relevantes"]:
        return "generator"
    if estado["reformulaciones"] >= MAX_REFORMULACIONES:
        return "generator"  # generar de todos modos (degradado)
    return "reformular"
```

---

## Comparación de estrategias

| Criterio | Score | LLM |
|----------|-------|-----|
| Latencia | ~0ms | +N × latencia LLM |
| Costo | 0 | N × tokens |
| Precisión | Media | Alta |
| Funciona offline | Sí | No |
| Entiende paráfrasis | No | Sí |
