# Health Checks

## Liveness vs Readiness

| | Liveness | Readiness |
|--|---------|-----------|
| Endpoint | `GET /health` | `GET /ready` |
| Propósito | ¿Está vivo el proceso? | ¿Puede servir tráfico? |
| Comprueba | Nada (solo responde) | ChromaDB, Redis, API keys |
| Si falla | Kubernetes reinicia el pod | Pod sale del load balancer |
| Frecuencia | Alta (cada 5s) | Media (cada 30s) |
| Latencia objetivo | < 10ms | < 2s |

---

## Liveness — trivial por diseño

```python
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}
```

No comprueba dependencias externas. Solo confirma que el proceso está vivo y acepta conexiones. Si tarda o falla → Kubernetes reinicia el pod.

---

## Readiness — verifica cada dependencia

```python
@app.get("/ready")
def ready():
    checks = {}
    overall = "ok"

    # ChromaDB — crítico (down si falla)
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        checks["chromadb"] = {"status": "ok", "detail": f"{count} docs"}
    except Exception as e:
        checks["chromadb"] = {"status": "down", "detail": str(e)}
        overall = "down"

    # Redis — opcional (degraded si falla, no down)
    try:
        redis.from_url(REDIS_URL).ping()
        checks["redis"] = {"status": "ok"}
    except:
        checks["redis"] = {"status": "degraded"}
        if overall == "ok":
            overall = "degraded"

    return {"status": overall, "checks": checks}
```

---

## Estados de salud

| Estado | HTTP | Significado |
|--------|------|-------------|
| `ok` | 200 | Todo funciona |
| `degraded` | 200 | Funciona con capacidades reducidas (sin cache) |
| `down` | 503 | No puede procesar requests |

Redis es `degraded` (no `down`) porque la API funciona sin él — solo pierde el cache.

---

## Kubernetes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 30
  failureThreshold: 3
```

## Docker Compose

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```
