# Nodo Hallucination Check — ¿La respuesta es fiel al contexto?

## ¿Qué es una alucinación en RAG?

El LLM genera información que NO está en los documentos recuperados:
- Inventa datos, fechas o nombres
- Hace inferencias que el contexto no respalda
- Usa conocimiento general cuando debería usar solo los docs

---

## LLM judge

```python
HALLUCINATION_PROMPT = ChatPromptTemplate.from_template(
    """¿La respuesta está completamente respaldada por el contexto?

    CONTEXTO: {contexto}
    RESPUESTA: {respuesta}

    VEREDICTO: fiel  (si TODO está en el contexto)
    VEREDICTO: alucinacion  (si hay algo inventado)
    RAZON: [explica en una oración]"""
)

def nodo_hallucination_check(estado):
    evaluacion = (HALLUCINATION_PROMPT | llm).invoke({
        "contexto": estado["contexto"][:2000],
        "respuesta": estado["respuesta"],
    }).content

    es_fiel = "alucinacion" not in evaluacion.lower()
    razon = extraer_linea(evaluacion, "RAZON:")

    return {"es_fiel": es_fiel, "razon_fidelidad": razon}
```

---

## Edge condicional post-check

```python
MAX_REGENERACIONES = 2

def decidir_post_check(estado) -> Literal["generator", END]:
    if estado["es_fiel"]:
        return END
    if estado["regeneraciones"] >= MAX_REGENERACIONES:
        return END  # aceptar aunque no sea perfecta
    return "generator"  # loop de regeneración
```

---

## ¿Por qué limitar las regeneraciones?

Sin límite, el grafo puede quedar en un loop infinito si el LLM consistentemente alucina. `MAX_REGENERACIONES = 2` garantiza terminación.

En la práctica, si después de 2 intentos sigue alucinando, el problema es el contexto (docs mal recuperados), no la generación.

---

## Hallucination check vs Grader

| | Grader | Hallucination Check |
|-|--------|---------------------|
| Qué evalúa | ¿Los docs responden la pregunta? | ¿La respuesta viene de los docs? |
| Cuándo actúa | Antes de generar | Después de generar |
| Si falla | Reformula la pregunta | Regenera la respuesta |

Son complementarios: el Grader previene generar con mala materia prima; el Hallucination Check verifica el producto final.

---

## Costo del pattern

Cada evaluación de hallucination es una llamada extra al LLM. En el peor caso (2 regeneraciones + 2 checks) son 4 llamadas extra por request. Considera:
- Usar un modelo más pequeño para el judge
- Cachear resultados del judge para preguntas similares
- Desactivarlo en entornos de bajo riesgo
