# Streaming Endpoint — SSE

## ¿Por qué streaming?

Sin streaming el usuario espera en silencio hasta que el LLM termina de generar. Con streaming:
- **Percepción de velocidad**: los tokens aparecen mientras se generan
- **UX familiar**: igual que ChatGPT, Claude, etc.
- **Cancelación temprana**: el usuario puede parar si la respuesta no va bien

---

## SSE vs WebSocket

| | SSE | WebSocket |
|--|---|---|
| Dirección | Solo servidor → cliente | Bidireccional |
| Complejidad | Baja | Alta |
| Retry automático | ✓ (nativo HTTP) | Manual |
| Caso de uso RAG | ✓ (respuesta unidireccional) | Overkill |

SSE es suficiente para RAG — el servidor envía tokens, el cliente solo escucha.

---

## Formato SSE

```
data: {"type": "sources", "sources": ["doc.pdf"]}\n\n
data: {"type": "token", "content": "La "}\n\n
data: {"type": "token", "content": "respuesta"}\n\n
data: {"type": "done"}\n\n
```

Cada evento: `data: <json>\n\n` — el doble newline cierra el evento.

---

## FastAPI StreamingResponse

```python
async def sse_generator(question: str):
    docs = retriever.invoke(question)
    prompt = QUERY_PROMPT.format_messages(
        context=format_docs(docs), question=question
    )
    async for chunk in llm.astream(prompt):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    return StreamingResponse(
        sse_generator(req.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # crítico si hay nginx por delante
        },
    )
```

`X-Accel-Buffering: no` es crítico — sin él nginx bufferiza toda la respuesta y el streaming no funciona.

---

## LangChain async streaming

```python
# Token por token
async for chunk in llm.astream(messages):
    print(chunk.content, end="", flush=True)

# El último chunk tiene usage_metadata (tokens consumidos)
async for chunk in llm.astream(messages):
    if chunk.usage_metadata:
        print(chunk.usage_metadata)  # {"input_tokens": 150, "output_tokens": 80}
```

---

## Cliente JavaScript

```javascript
const response = await fetch('/query/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({question: 'texto'}),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split('\n\n');
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      if (event.type === 'token') output += event.content;
    }
  }
}
```
