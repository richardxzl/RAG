# ¿Qué es un Agente?

## La diferencia fundamental

```
Chain:  input → paso1 → paso2 → paso3 → output
        (flujo hardcoded, siempre igual)

Agente: input → LLM decide → acción → observación → LLM decide → ... → output
        (el LLM controla el flujo en tiempo de ejecución)
```

En una **chain**, el programador decide el orden de pasos en el código.
En un **agente**, el **LLM** decide qué hacer, en qué orden, y si necesita más pasos.

---

## ¿Cómo decide el agente?

El mecanismo es **tool calling** (function calling):

1. Le das al LLM una lista de herramientas con sus descripciones
2. El LLM analiza la pregunta y genera un JSON con la tool que quiere llamar
3. Tu código ejecuta la tool y le devuelve el resultado
4. El LLM procesa el resultado y decide si necesita más tools o responde

```python
# El LLM genera esto internamente:
{
  "name": "calcular",
  "args": {"expresion": "347 * 28"}
}
# Tu código lo ejecuta y devuelve: "347 * 28 = 9716"
# El LLM procesa ese resultado y formula la respuesta final
```

---

## Definir herramientas

```python
from langchain_core.tools import tool

@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática. Ejemplo: '2 + 2', '10 * 5'."""
    resultado = eval(expresion, {"__builtins__": {}}, {})
    return f"{expresion} = {resultado}"
```

El **docstring** es lo que el LLM lee para decidir cuándo usar la tool.
Debe ser claro y específico.

---

## Crear un agente (forma simple)

```python
from langgraph.prebuilt import create_react_agent

llm = get_llm()
tools = [calcular, buscar_definicion, contar_palabras]
agente = create_react_agent(llm, tools)

resultado = agente.invoke({
    "messages": [HumanMessage(content="¿Cuánto es 15 * 8?")]
})
respuesta = resultado["messages"][-1].content
```

---

## ¿Cuándo usar agente vs chain?

| Usar chain cuando... | Usar agente cuando... |
|---------------------|-----------------------|
| El flujo es siempre igual | El flujo depende de la pregunta |
| Latencia crítica | Precisión importa más que velocidad |
| Sin herramientas externas | Necesitas herramientas externas |
| Flujo predecible y auditable | El LLM sabe mejor qué hacer |

---

## El grafo interno del agente

LangGraph representa el agente ReAct como un grafo de dos nodos en loop:

```
START → agent_node ← ──────────────────┐
             ↓                          │
     ¿tiene tool_calls?                 │
         ↓ Sí          ↓ No            │
    tools_node        END               │
         └──────────────────────────────┘
```

`create_react_agent` construye exactamente este grafo.
