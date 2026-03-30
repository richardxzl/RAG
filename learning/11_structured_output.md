# StructuredOutputParser — Forzar respuestas JSON

## El problema

El LLM retorna strings. Tu aplicación necesita datos: dicts, listas, números. Sin un parser, tienes que hacer `json.loads()` manualmente, manejar texto extra que el LLM añade, y lidiar con keys inconsistentes.

```python
# Sin parser — frágil
respuesta = chain.invoke({"resena": "..."})
# respuesta puede ser:
# '{"sentimiento": "positivo"}'          ← ideal, rara vez pasa
# 'Aquí está el análisis:\n{"sent...}'   ← texto extra antes del JSON
# "El sentimiento es **positivo**..."    ← sin JSON en absoluto
```

`StructuredOutputParser` resuelve esto inyectando instrucciones de formato en el prompt y extrayendo el JSON de la respuesta automáticamente.

---

## ResponseSchema — definir los campos

Cada `ResponseSchema` define un campo del dict resultante:

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

schemas = [
    ResponseSchema(
        name="sentimiento",
        description="Sentimiento general: 'positivo', 'negativo' o 'neutro'"
    ),
    ResponseSchema(
        name="puntuacion",
        description="Puntuación numérica del 1 al 5"
    ),
    ResponseSchema(
        name="aspectos_positivos",
        description="Lista de aspectos positivos. Array de strings."
    ),
]

parser = StructuredOutputParser.from_response_schemas(schemas)
```

**La `description` no es solo documentación** — el LLM la lee como instrucción. Una descripción vaga produce campos vagos.

---

## format_instructions — el puente con el prompt

```python
instructions = parser.get_format_instructions()
# Genera algo como:
# "Return a markdown code snippet with a JSON object formatted to look like:
# ```json
# {
#     "sentimiento": string  // Sentimiento general: 'positivo'...
#     "puntuacion": string   // Puntuación numérica del 1 al 5
# }
# ```"
```

Estas instrucciones **deben ir en el prompt** — normalmente en el `system` o como variable:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "Analiza la reseña.\n\n{format_instructions}"),
    ("human", "{resena}"),
])

chain = template | llm | parser  # parser como último paso

resultado = chain.invoke({
    "resena": "Excelente producto...",
    "format_instructions": parser.get_format_instructions(),
})
# resultado → {"sentimiento": "positivo", "puntuacion": "5", ...}
```

---

## Cómo funciona el parsing internamente

`StructuredOutputParser.parse()` busca el bloque de código markdown con JSON en la respuesta del LLM:

```
```json
{"sentimiento": "positivo", "puntuacion": "4"}
```
```

Si no lo encuentra, lanza `OutputParserException`. No valida tipos — todo llega como string.

---

## Limitaciones

| Limitación | Detalle |
|-----------|---------|
| Sin validación de tipos | `"puntuacion"` llega como `"4"` (string), no `4` (int) |
| Frágil con modelos menores | Modelos pequeños ignoran las format_instructions |
| No hay schema estricto | No puedes definir que un campo sea `Optional` o tenga `Literal` |
| Parsing frágil | Si el LLM añade texto fuera del bloque JSON, puede fallar |

Para validación tipada real, usa `PydanticOutputParser` (módulo 2.4).

---

## Cuándo usarlo

| Situación | StructuredOutputParser | PydanticOutputParser |
|-----------|----------------------|---------------------|
| Prototipo rápido | ✅ Menos setup | — |
| Campos simples, todos strings | ✅ | — |
| Necesitas tipos (int, float, list) | ❌ | ✅ |
| Validación de valores | ❌ | ✅ |
| Modelo anidado (objeto dentro de objeto) | ❌ Complejo | ✅ Nativo |

---

## Regla práctica

> `StructuredOutputParser` es el nivel de entrada. Si te basta con un dict de strings y el modelo que usas lo sigue razonablemente bien, úsalo. En cuanto necesites tipos, validación, o modelos anidados, sube a `PydanticOutputParser`.
