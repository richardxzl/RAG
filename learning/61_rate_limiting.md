# Rate Limiting

## ¿Por qué limitar?

Cada request al RAG cuesta dinero (LLM de Anthropic), tiempo y recursos. Sin rate limiting un solo cliente malicioso puede:
- Agotar el budget de la API key
- Saturar el servidor con requests concurrentes
- Degradar el servicio para todos los demás usuarios

---

## slowapi — el estándar para FastAPI

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/query")
@limiter.limit("10/minute")
def query(req: QueryRequest, request: Request):
    ...

@app.post("/query/premium")
@limiter.limit("100/minute")  # tier premium
def query_premium(req: QueryRequest, request: Request):
    ...
```

Formatos de límite: `"10/minute"`, `"100/hour"`, `"1000/day"`, `"5/second"`.

---

## Claves de rate limiting

| `key_func` | Uso |
|---------|-----|
| `get_remote_address` | Por IP — protección básica |
| API key en header | Por cliente autenticado |
| User ID en JWT | Por usuario |
| Función custom | Cualquier combinación |

---

## Middleware manual (sin dependencias)

Ventana deslizante en memoria:

```python
_requests: dict[str, list[float]] = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host
    now = time()
    _requests[ip] = [t for t in _requests[ip] if now - t < WINDOW]
    if len(_requests[ip]) >= LIMIT:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers={"Retry-After": str(int(WINDOW))},
        )
    _requests[ip].append(now)
    return await call_next(request)
```

**Limitación**: el estado está en memoria — no se comparte entre workers ni persiste entre reinicios. Para producción con múltiples instancias: usar Redis como backend.

---

## Respuesta 429

```json
{"error": "Rate limit exceeded", "limit": 10, "window_seconds": 60}
```

Header: `Retry-After: 60` (segundos hasta el próximo intento disponible).
