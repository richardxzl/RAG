# Memoria Conversacional en RAG

## El problema

Sin memoria, cada pregunta al chat es independiente. Si preguntas "¿Qué dice sobre microservicios?" y luego "¿Y sobre su escalabilidad?", el "su" no tiene referencia — el LLM no sabe que "su" se refiere a microservicios.

## ¿Cómo funciona la memoria?

Se inyecta el historial de mensajes anteriores en el prompt, para que el LLM tenga contexto de la conversación.

```
[System] Eres un asistente...
         Contexto: {chunks recuperados}

[Human] ¿Qué dice sobre microservicios?        ← turno 1
[AI]    El documento menciona que...             ← turno 1

[Human] ¿Y sobre su escalabilidad?              ← turno 2 (AHORA el LLM sabe que "su" = microservicios)
```

## Estrategias de memoria

### 1. Buffer Window (la que usamos)

Guarda los últimos N mensajes tal cual.

```python
chat_history = []
# Después de cada turno:
chat_history.append(HumanMessage(content=question))
chat_history.append(AIMessage(content=answer))
# Al invocar, solo pasamos los últimos N:
chat_history[-MEMORY_WINDOW:]
```

**Pros:** Simple, predecible, sin tokens extra.
**Contras:** Pierde todo lo anterior a la ventana. No escala si los mensajes son largos.

### 2. Summary Memory

Un LLM resume el historial en un párrafo. En vez de pasar 20 mensajes, pasas un resumen.

**Pros:** Escala a conversaciones largas.
**Contras:** Cuesta tokens generar el resumen. Puede perder detalles.

### 3. Buffer + Summary (híbrido)

Los últimos N mensajes van completos, los anteriores se resumen.

**Pros:** Balance entre detalle reciente y contexto histórico.
**Contras:** Más complejo de implementar.

### 4. Vector Memory

Cada mensaje se convierte en embedding y se busca por relevancia (no por recencia).

**Pros:** Recupera contexto relevante sin importar cuándo se dijo.
**Contras:** Puede traer mensajes irrelevantes temporalmente.

## Implementación con LCEL

La clave es `MessagesPlaceholder` en el prompt:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "Contexto: {context}"),
    MessagesPlaceholder("chat_history"),  # ← aquí se inyecta el historial
    ("human", "{question}"),
])
```

Y al invocar:

```python
answer = chain.invoke({
    "question": "mi pregunta",
    "chat_history": chat_history[-6:],  # últimos 6 mensajes
})
```

## Consideraciones importantes

### Tokens = dinero

Cada mensaje en el historial consume tokens de input. Con Haiku es barato, pero en producción con Opus, 20 turnos de conversación pueden ser significativos.

### Memoria NO es igual a historial de chat

- **Historial** = todos los mensajes raw.
- **Memoria** = lo que decides INYECTAR en el prompt. Puedes filtrar, resumir, o seleccionar.

### Memoria + RAG = más contexto en el prompt

Tu prompt ahora tiene: system prompt + contexto RAG + historial + pregunta. Si cada chunk tiene 1000 chars y traes 4, más 6 mensajes de historial, el prompt puede crecer bastante. Monitorea el uso de tokens.

### ¿Cuántos turnos guardar?

| Caso de uso | Window recomendado |
|-------------|-------------------|
| FAQ bot (preguntas independientes) | 0-2 mensajes |
| Chat de soporte | 4-8 mensajes |
| Asistente de investigación | 8-12 mensajes o summary |
| Agente con tareas multi-paso | Summary + vector |

## Qué sigue

- **ConversationSummaryMemory** — Para conversaciones largas donde la ventana no alcanza.
- **Persistencia** — Guardar el historial en DB para retomar conversaciones.
- **LangGraph** — Memoria como estado del grafo, con control total sobre qué se persiste.
