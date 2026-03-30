# Tool Calling — El LLM decide qué herramientas usar

## ¿Qué es tool calling?

Es el mecanismo que permite al LLM generar **llamadas a funciones estructuradas** en vez de texto libre. El LLM no ejecuta la función — genera un JSON que describe qué función llamar y con qué argumentos.

---

## Flujo completo

```python
# 1. Enlazar tools al LLM
llm_con_tools = llm.bind_tools([calcular, buscar_definicion])

# 2. El LLM decide si necesita una tool
respuesta: AIMessage = llm_con_tools.invoke(mensajes)

# 3. Inspeccionar si hay tool calls
if respuesta.tool_calls:
    for tc in respuesta.tool_calls:
        # tc = {"id": "...", "name": "calcular", "args": {"expresion": "5*5"}}
        resultado = ejecutar_tool(tc)

        # 4. Devolver resultado al LLM
        mensajes.append(ToolMessage(
            content=resultado,
            tool_call_id=tc["id"],
        ))

# 5. El LLM genera la respuesta final con los resultados
respuesta_final = llm_con_tools.invoke(mensajes)
```

---

## Lo que genera el LLM

```json
[
  {
    "id": "toolu_01ABC...",
    "name": "calcular",
    "args": {
      "expresion": "347 * 28"
    }
  }
]
```

Este JSON está en `AIMessage.tool_calls`. El LLM nunca ejecuta el código — solo genera la descripción.

---

## Definir herramientas

### Con @tool (más simple)

```python
@tool
def buscar(query: str) -> str:
    """Busca información sobre un tema. Usa esto para preguntas de conocimiento."""
    return hacer_busqueda(query)
```

### Con StructuredTool (más control)

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

class BuscarInput(BaseModel):
    query: str
    max_resultados: int = 5

def buscar_impl(query: str, max_resultados: int = 5) -> str:
    return hacer_busqueda(query, max_resultados)

buscar = StructuredTool.from_function(
    func=buscar_impl,
    name="buscar",
    description="Busca información sobre un tema.",
    args_schema=BuscarInput,
)
```

---

## ToolMessage — el resultado de la tool

```python
ToolMessage(
    content="el resultado de la tool como string",
    tool_call_id=tc["id"],   # debe coincidir con el id del tool_call
)
```

El `tool_call_id` es la clave: el LLM relaciona el resultado con la llamada que hizo.

---

## Múltiples tools en paralelo

El LLM puede solicitar múltiples tool calls en un solo `AIMessage.tool_calls`.
El framework las puede ejecutar en paralelo:

```python
# AIMessage.tool_calls puede tener más de una entrada:
[
  {"name": "buscar_poblacion", "args": {"ciudad": "Madrid"}},
  {"name": "buscar_poblacion", "args": {"ciudad": "Barcelona"}},
]
# Ejecutar ambas en paralelo y devolver dos ToolMessages
```

`ToolNode` de LangGraph maneja esto automáticamente.
