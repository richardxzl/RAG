# Loop: Reformular → Reintentar — Corrective RAG completo

## El patrón completo

Corrective RAG combina todos los nodos del módulo 8 en un sistema que se auto-corrige:

```
START
  ↓
Router ──────────────────────────────── directo → END
  ↓ (rag)
Retriever
  ↓
Grader ── relevantes → Generator → Hallucination Check ── fiel → END
  ↓ (no relevantes)                         ↓ (no fiel, < max)
Reformular ←──────────────────── vuelve a Generator
  ↓
vuelve a Retriever (max 2 veces)
```

---

## Estado centralizado

```python
class EstadoCorrectiveRAG(TypedDict):
    pregunta_original: str       # nunca cambia
    pregunta_actual: str         # puede cambiar en cada reformulación
    ruta: str                    # "rag" o "conversacional"
    documentos: list[Document]
    documentos_relevantes: list[Document]
    contexto: str
    respuesta: str
    es_fiel: bool
    reformulaciones: int         # contador de reformulaciones
    regeneraciones: int          # contador de regeneraciones
    logs: Annotated[list[str], operator.add]
```

---

## Nodo Reformular

```python
REFORMULAR_PROMPT = ChatPromptTemplate.from_template(
    """La pregunta no encontró documentos relevantes.
    Reformúlala con otras palabras, mismo significado.

    Pregunta original: {pregunta}
    Intento número: {intento}

    Nueva pregunta:"""
)

def nodo_reformular(estado):
    nueva_pregunta = (REFORMULAR_PROMPT | llm).invoke({
        "pregunta": estado["pregunta_actual"],
        "intento": estado["reformulaciones"] + 1,
    }).content.strip()

    return {
        "pregunta_actual": nueva_pregunta,
        "reformulaciones": estado["reformulaciones"] + 1,
        "documentos": [],            # limpiar para el siguiente ciclo
        "documentos_relevantes": [],
    }
```

El nodo limpia `documentos` y `documentos_relevantes` para que el próximo Retriever empiece fresh.

---

## Los dos loops

```python
# Loop 1: reformulación (Grader → Reformular → Retriever)
builder.add_edge("reformular", "retriever")  # edge hacia atrás

# Loop 2: regeneración (Hallucination → Generator)
builder.add_conditional_edges("hallucination_check", decidir_post_check, {
    "generator": "generator",   # edge hacia atrás
    END: END,
})
```

---

## Contadores de ciclos

```python
MAX_REFORMULACIONES = 2
MAX_REGENERACIONES = 2

def grader_decision(estado):
    if estado["documentos_relevantes"]:
        return "generator"
    if estado["reformulaciones"] >= MAX_REFORMULACIONES:
        return "generator"  # aceptar lo que hay
    return "reformular"

def hallucination_decision(estado):
    if estado["es_fiel"]:
        return END
    if estado["regeneraciones"] >= MAX_REGENERACIONES:
        return END          # aceptar aunque no sea perfecta
    return "generator"
```

Los contadores garantizan **terminación garantizada**. Sin ellos, un loop puede ejecutarse indefinidamente.

---

## Responder la pregunta original

Aunque la pregunta se reformule para mejorar el retrieval, el Generator siempre responde la pregunta original:

```python
respuesta = (PROMPT | llm).invoke({
    "context": contexto,
    "question": estado["pregunta_original"],  # no pregunta_actual
})
```

---

## Casos de uso de Corrective RAG

- Documentos técnicos con jerga específica (la reformulación usa sinónimos)
- Preguntas ambiguas que el usuario expresa de forma imprecisa
- Bases de conocimiento grandes donde la primera búsqueda a veces falla
- Sistemas de alta confianza (médico, legal) donde la fidelidad importa
