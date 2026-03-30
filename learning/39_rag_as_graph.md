# RAG como Grafo — Migrar de LCEL a LangGraph

## ¿Por qué migrar?

El RAG en LCEL es un pipe lineal que funciona bien para el caso simple. La limitación aparece cuando necesitas:
- Inspeccionar los documentos recuperados antes de generar
- Agregar una capa de validación entre el retriever y el generator
- Implementar un loop de reformulación si los docs no son relevantes

```
LCEL:       retriever | format_docs | prompt | llm | parser
LangGraph:  retriever_node → generator_node  (con estado compartido)
```

---

## El Estado del RAG

```python
class EstadoRAG(TypedDict):
    pregunta: str
    documentos: list[Document]    # escrito por retriever, leído por generator
    contexto: str                 # escrito por generator
    respuesta: str
    logs: Annotated[list[str], operator.add]
```

El estado centraliza los datos. El generator ya no necesita recibir los docs por parámetro — los lee del estado que el retriever escribió.

---

## Nodo Retriever

```python
def nodo_retriever(estado: EstadoRAG) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(estado["pregunta"])
    return {
        "documentos": docs,
        "logs": [f"retriever: {len(docs)} docs"],
    }
```

---

## Nodo Generator

```python
def nodo_generator(estado: EstadoRAG) -> dict:
    contexto = format_docs(estado["documentos"])
    respuesta = (QUERY_PROMPT | llm).invoke({
        "context": contexto,
        "question": estado["pregunta"],
    }).content
    return {"contexto": contexto, "respuesta": respuesta}
```

---

## Grafo

```python
builder = StateGraph(EstadoRAG)
builder.add_node("retriever", nodo_retriever)
builder.add_node("generator", nodo_generator)
builder.add_edge(START, "retriever")
builder.add_edge("retriever", "generator")
builder.add_edge("generator", END)
grafo = builder.compile()
```

---

## Qué cambia vs LCEL

| Aspecto | LCEL | LangGraph |
|---------|------|-----------|
| Datos intermedios | No accesibles | `estado["documentos"]` |
| Testing nodos | Difícil aislar | Función pura testeable |
| Agregar validación | Modificar el pipe | Agregar un nodo |
| Loops | Imposible | Edges condicionales de vuelta |
