# FastAPI Wrapper del RAG

## ¿Por qué FastAPI?

El RAG como script Python solo puede usarse en local. Envolverlo en una API REST permite que cualquier cliente (web, móvil, otro microservicio) lo consuma independientemente del lenguaje.

FastAPI añade sobre Flask/Django:
- **Tipos automáticos**: Pydantic valida requests/responses y genera la documentación
- **Async nativo**: compatible con `astream`, `ainvoke` de LangChain
- **OpenAPI gratis**: `/docs` (Swagger UI) y `/redoc` sin configuración

---

## Inicialización de la chain

**Fuera del endpoint** — se ejecuta una sola vez al arrancar el proceso:

```python
# ✓ Una sola vez al arrancar
_query_fn, _cache = build_query_chain()

@app.post("/query")
def query(req: QueryRequest):
    return _query_fn(req.question)  # reutiliza la chain caliente
```

Si la inicializas dentro del endpoint, cada request carga los embeddings, conecta a ChromaDB y construye el LLM desde cero — cold start de ~2s por request.

---

## Modelos Pydantic

```python
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    cache_hit: bool
```

FastAPI valida automáticamente los tipos. Si `question` falta o no es string → 422 Unprocessable Entity.

---

## TestClient — demo sin servidor

`TestClient` de FastAPI permite llamar endpoints directamente en el mismo proceso, sin abrir un puerto TCP. Útil para scripts de demo y tests:

```python
from fastapi.testclient import TestClient

client = TestClient(app)
resp = client.post("/query", json={"question": "¿Qué es LangChain?"})
data = resp.json()  # igual que con httpx/requests real
```

---

## Arrancar el servidor

```bash
# Desarrollo — recarga automática al editar
uvicorn 59_fastapi_wrapper:app --reload --port 8000

# Producción — múltiples workers
uvicorn 59_fastapi_wrapper:app --workers 4 --port 8000

# Con gunicorn (más estable en prod)
gunicorn 59_fastapi_wrapper:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## Documentación automática

- `http://localhost:8000/docs` — Swagger UI interactivo (probar endpoints en el navegador)
- `http://localhost:8000/redoc` — ReDoc
- `http://localhost:8000/openapi.json` — Schema JSON para integraciones
