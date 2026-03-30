# Tools Custom — RAG, web, cálculos

## Anatomy de una tool

```python
from langchain_core.tools import tool

@tool
def mi_tool(argumento: str, otro: int = 5) -> str:
    """
    Descripción clara de qué hace y cuándo usarla.
    El LLM lee este docstring para decidir si invocar esta tool.
    Sé específico: menciona casos de uso y limitaciones.
    """
    resultado = hacer_algo(argumento, otro)
    return str(resultado)   # siempre retorna string
```

---

## Reglas de diseño

**El docstring ES la descripción que ve el LLM.** Un docstring vago produce decisiones de routing malas.

```python
# Malo: el LLM no sabe cuándo usar esto
@tool
def buscar(q: str) -> str:
    """Busca cosas."""
    ...

# Bueno: el LLM sabe exactamente cuándo usarla
@tool
def buscar_en_documentos(pregunta: str) -> str:
    """
    Busca información en la base de conocimiento del proyecto.
    Úsala cuando el usuario pregunta sobre el contenido de documentos específicos
    o necesita información de los archivos cargados. NO uses para conocimiento general.
    """
    ...
```

---

## Tool RAG

```python
@tool
def buscar_en_documentos(pregunta: str) -> str:
    """Busca información en los documentos del proyecto."""
    retriever = get_retriever()
    docs = retriever.invoke(pregunta)
    if not docs:
        return "No encontré documentos relevantes."
    return format_docs(docs[:3])[:1500]  # limitar tokens
```

Convertir el RAG en una tool permite al agente decidir **cuándo** buscar en documentos — no siempre para toda pregunta.

---

## Tool con estado (closure)

Las tools son stateless por defecto. Para estado, usa closure:

```python
def crear_tools_con_historial():
    historial = []   # estado compartido entre las dos tools

    @tool
    def recordar(dato: str) -> str:
        """Guarda un dato para más tarde."""
        historial.append(dato)
        return f"Guardado: '{dato}'"

    @tool
    def listar_recordados() -> str:
        """Lista los datos guardados."""
        return "\n".join(historial) or "Nada guardado aún."

    return recordar, listar_recordados
```

---

## Tool con validación (Pydantic)

```python
from pydantic import BaseModel, validator

class ConvertirInput(BaseModel):
    valor: float
    de: str
    a: str

    @validator("de", "a")
    def moneda_valida(cls, v):
        monedas = ["USD", "EUR", "MXN"]
        if v.upper() not in monedas:
            raise ValueError(f"Moneda no soportada: {v}")
        return v.upper()

@tool(args_schema=ConvertirInput)
def convertir(valor: float, de: str, a: str) -> str:
    """Convierte entre monedas: USD, EUR, MXN."""
    ...
```

---

## Checklist de una buena tool

- [ ] Docstring explica cuándo usarla (no solo qué hace)
- [ ] Args con nombres descriptivos y type hints
- [ ] Maneja errores y retorna strings informativos (no lanza excepciones)
- [ ] Retorna solo lo necesario (no saturar el contexto del LLM)
- [ ] Nombre único, snake_case, verbo + sustantivo
