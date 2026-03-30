# Nodo Router — ¿RAG o conversacional?

## El problema

Sin router, cada pregunta activa el retriever aunque no necesite documentos. "Hola", "gracias" y preguntas de conocimiento general van innecesariamente al vectorstore.

---

## Estrategia 1: Keywords (sin LLM)

```python
KEYWORDS_RAG = ["documento", "archivo", "texto", "según", "menciona", "dice"]
KEYWORDS_CONVERSACIONAL = ["hola", "gracias", "cómo estás", "quién eres"]

def nodo_router_keywords(estado):
    pregunta = estado["pregunta"].lower()
    if any(k in pregunta for k in KEYWORDS_CONVERSACIONAL):
        ruta = "conversacional"
    elif any(k in pregunta for k in KEYWORDS_RAG):
        ruta = "rag"
    else:
        ruta = "rag"  # default seguro
    return {"ruta": ruta}
```

**Ventajas**: cero latencia, cero costo, predecible.
**Desventajas**: no entiende paráfrasis, fácil de engañar.

---

## Estrategia 2: LLM judge

```python
ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """Clasifica la pregunta. Responde SOLO con una palabra.
    rag → pregunta sobre el contenido de un documento
    conversacional → saludo o pregunta genérica

    Pregunta: {pregunta}
    Respuesta:"""
)

def nodo_router_llm(estado):
    resultado = (ROUTER_PROMPT | llm).invoke({"pregunta": estado["pregunta"]}).content
    ruta = "rag" if "rag" in resultado.lower() else "conversacional"
    return {"ruta": ruta}
```

**Ventajas**: entiende contexto y variaciones del lenguaje.
**Desventajas**: +latencia y costo por cada request.

---

## Edge condicional post-router

```python
def decidir_ruta(estado) -> Literal["retriever", "directo"]:
    return "retriever" if estado["ruta"] == "rag" else "directo"

builder.add_conditional_edges("router", decidir_ruta, {
    "retriever": "retriever",
    "directo": "directo",
})
```

---

## ¿Cuándo usar cada estrategia?

| Criterio | Keywords | LLM |
|----------|----------|-----|
| Volumen alto | ✓ | — |
| Preguntas predecibles | ✓ | — |
| Lenguaje variado | — | ✓ |
| Multilingüe | — | ✓ |

En producción: combinar ambos. Keywords como fast-path, LLM como fallback para los casos dudosos.
