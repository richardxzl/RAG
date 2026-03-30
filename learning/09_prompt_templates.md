# Prompt Templates — PromptTemplate vs ChatPromptTemplate

## El problema que resuelven

Un prompt no es solo un string. Tiene estructura: rol del sistema, historial de conversación, input del usuario, variables dinámicas. Construirlo con concatenación de strings es frágil y no escala.

Los templates formalizan esa estructura y la hacen componible con el resto del pipeline LCEL.

---

## PromptTemplate — completion style

Para modelos de texto plano (GPT-3, modelos locales de completion). Genera un string.

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Sugiere una receta para: {ingrediente}. Máximo 3 oraciones."
)

# Ver el string resultante antes de enviarlo al LLM
print(template.format(ingrediente="tomates"))
# → "Sugiere una receta para: tomates. Máximo 3 oraciones."

# Usar en pipeline
chain = template | llm | StrOutputParser()
```

**Cuándo todavía tiene sentido**: modelos open source corriendo localmente con `Ollama` o similar que no tienen el concepto de roles. Para todo lo demás, usa `ChatPromptTemplate`.

---

## ChatPromptTemplate — chat style

Para todos los chat models modernos (Claude, GPT-4, Gemini). Genera una lista de `BaseMessage`.

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente experto en cocina. Responde en español."),
    ("human", "Sugiere una receta para: {ingrediente}"),
])
```

Los roles disponibles son:

| Rol | Clase interna | Propósito |
|-----|---------------|-----------|
| `"system"` | `SystemMessage` | Instrucciones al LLM, tono, restricciones |
| `"human"` | `HumanMessage` | Input del usuario |
| `"ai"` | `AIMessage` | Respuesta previa del LLM (para historial) |

### from_template() vs from_messages()

```python
# from_template → crea un único HumanMessage (sin system)
ChatPromptTemplate.from_template("Pregunta: {q}")

# from_messages → control total sobre los roles
ChatPromptTemplate.from_messages([
    ("system", "..."),
    ("human", "{q}"),
])
```

Usa `from_template()` solo para prototipos rápidos. En producción, `from_messages()` te da control explícito sobre qué le dices al LLM como instrucción vs como input.

---

## Inspeccionar un template

```python
print(template.input_variables)   # ['ingrediente']
print(template.messages)          # lista de MessagePromptTemplate

# Formatear sin invocar el LLM
mensajes = template.format_messages(ingrediente="tomates")
for msg in mensajes:
    print(f"[{type(msg).__name__}] {msg.content}")
```

Formatear sin invocar es útil para debugear — ver exactamente qué recibe el LLM.

---

## partial() — variables pre-fijadas

Cuando tienes un template con 3 variables pero una siempre vale lo mismo, no la repitas en cada `.invoke()`:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "Responde en {idioma} con tono {tono}."),
    ("human", "{pregunta}"),
])

# Fijar idioma y tono una vez
template_es = template.partial(idioma="español", tono="formal")

# Ahora solo necesitas pasar pregunta
chain = template_es | llm | StrOutputParser()
chain.invoke({"pregunta": "¿Qué es la fotosíntesis?"})
```

El template original no se modifica. `.partial()` retorna un nuevo template con las variables fijadas.

---

## MessagesPlaceholder — historial dinámico

Para insertar una lista variable de mensajes (historial de conversación) en un punto del template:

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

template = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente conversacional."),
    MessagesPlaceholder("chat_history"),   # ← lista de mensajes aquí
    ("human", "{pregunta}"),
])

# Primer turno: historial vacío
chain.invoke({"chat_history": [], "pregunta": "Hola, me llamo Ana."})

# Segundo turno: incluir el intercambio anterior
chain.invoke({
    "chat_history": [
        HumanMessage(content="Hola, me llamo Ana."),
        AIMessage(content="Hola Ana, ¿en qué puedo ayudarte?"),
    ],
    "pregunta": "¿Cómo me llamo?"
})
```

---

## Tradeoffs

| | PromptTemplate | ChatPromptTemplate |
|---|---|---|
| **Simplicidad** | Mayor — un string | Menor — lista de mensajes |
| **Control** | Ninguno sobre roles | Total sobre system/human/ai |
| **Compatibilidad** | Completion + chat models | Solo chat models |
| **Historial** | Manual (concatenar) | Nativo con MessagesPlaceholder |
| **Recomendado para** | Modelos locales legacy | Todo lo demás |

---

## Regla práctica

> Usa `ChatPromptTemplate.from_messages()` siempre que trabajes con un chat model. El rol `system` es la forma más efectiva de dar instrucciones al LLM — no lo desperdicies metiéndolo en el turno `human`.
