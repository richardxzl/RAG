# PydanticOutputParser — Validación tipada de la salida

## Por qué Pydantic > StructuredOutputParser

`StructuredOutputParser` retorna un `dict` donde todos los valores son strings. No hay validación: si el LLM pone `"relevancia": "alta"` cuando esperabas un entero, tu código recibe un string y falla más tarde.

`PydanticOutputParser` retorna un **objeto Pydantic validado**. Si el tipo no coincide, falla inmediatamente con un error claro.

```python
# StructuredOutputParser
resultado["relevancia"]  # "8" — string siempre

# PydanticOutputParser
resultado.relevancia     # 8 — int validado, con IDE autocomplete
```

---

## Definir el modelo

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal

class NoticiaEstructurada(BaseModel):
    titulo: str = Field(description="Título en máximo 10 palabras")
    categoria: Literal["tecnología", "economía", "política", "otro"] = Field(
        description="Categoría temática principal"
    )
    entidades: list[str] = Field(
        description="Personas, empresas u organizaciones mencionadas"
    )
    fecha_mencionada: Optional[str] = Field(
        default=None,
        description="Fecha mencionada, null si no hay ninguna"
    )
    relevancia: int = Field(ge=1, le=10, description="Relevancia del 1 al 10")
```

### Field(description=...) es una instrucción, no documentación

El LLM lee la `description` para saber qué poner en ese campo. Una descripción vaga produce resultados vagos.

```python
# ❌ Vago
relevancia: int = Field(description="Número")

# ✅ Instructivo
relevancia: int = Field(ge=1, le=10, description="Relevancia del 1 (baja) al 10 (alta) según el impacto")
```

---

## Tipos soportados

| Tipo Python | Efecto en el LLM |
|-------------|------------------|
| `str` | String libre |
| `int`, `float` | Número (el LLM intenta respetar el tipo) |
| `bool` | `true` / `false` |
| `list[str]` | Array JSON de strings |
| `Optional[str]` | String o `null` |
| `Literal["a", "b"]` | Solo acepta esos valores exactos |
| `BaseModel` anidado | Objeto JSON anidado |

Los constraints de Pydantic (`ge`, `le`, `min_length`, etc.) también se incluyen en el schema — el LLM los ve como restricciones.

---

## get_format_instructions() — el JSON schema

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=NoticiaEstructurada)
print(parser.get_format_instructions())
```

Genera el JSON schema del modelo Pydantic y lo envuelve en instrucciones para el LLM. Debes inyectarlo en el prompt:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "Extrae información de la noticia.\n\n{format_instructions}"),
    ("human", "{noticia}"),
])

chain = template | llm | parser

resultado = chain.invoke({
    "noticia": "...",
    "format_instructions": parser.get_format_instructions(),
})
# resultado es un objeto NoticiaEstructurada, no un dict
print(resultado.titulo)      # str
print(resultado.relevancia)  # int, validado
```

---

## Pydantic v1 vs v2

Este proyecto usa Pydantic v2. Diferencias clave:

| v1 | v2 |
|----|-----|
| `resultado.dict()` | `resultado.model_dump()` |
| `resultado.json()` | `resultado.model_dump_json()` |
| `Model.parse_raw(json)` | `Model.model_validate_json(json)` |
| `@validator` | `@field_validator` |

LangChain soporta ambas versiones, pero v2 es más rápida y estricta por defecto.

---

## Error handling

```python
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException

try:
    resultado = parser.parse(respuesta_llm)
except OutputParserException as e:
    # El LLM no generó JSON válido
    print(f"JSON malformado: {e}")
except ValidationError as e:
    # JSON válido pero tipos incorrectos
    print(f"Validación fallida: {e}")
```

---

## PydanticOutputParser vs .with_structured_output()

`with_structured_output()` es la alternativa moderna que usa **tool calling** nativo del modelo:

```python
# Con PydanticOutputParser (prompt engineering)
chain = template | llm | parser

# Con .with_structured_output() (tool calling nativo)
llm_structured = llm.with_structured_output(NoticiaEstructurada)
resultado = llm_structured.invoke("Texto de la noticia...")
```

| | PydanticOutputParser | with_structured_output() |
|---|---|---|
| **Mecanismo** | Format instructions en el prompt | Tool calling nativo |
| **Confiabilidad** | Depende del LLM siguiendo instrucciones | Mayor — forzado por el modelo |
| **Compatibilidad** | Cualquier LLM | Solo modelos con tool calling (Claude, GPT-4, etc.) |
| **Setup** | Más verboso | Más simple |
| **Streams** | ✅ `.stream()` funciona | ⚠️ Limitado |

**Regla práctica**: si el modelo soporta tool calling, usa `.with_structured_output()`. `PydanticOutputParser` sirve para modelos locales o cuando necesitas control total sobre el prompt.
