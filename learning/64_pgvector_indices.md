# Índices en pgvector

## Sin índice = sequential scan

Sin índice, pgvector compara el vector query contra TODOS los vectores de la tabla (O(n)). Funciona para miles de documentos, pero con 100K+ la latencia se vuelve inaceptable.

---

## HNSW — el estándar actual

```sql
CREATE INDEX idx_embedding_hnsw
ON langchain_pg_embedding
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Ajustar precisión en búsqueda (ejecutar antes de cada query)
SET hnsw.ef_search = 100;
```

| Parámetro | Default | Efecto al aumentar |
|---------|---------|----------------|
| `m` | 16 | Más preciso, más RAM |
| `ef_construction` | 64 | Mejor índice, más lento al indexar |
| `ef_search` | 40 | Más preciso en búsqueda, más lento |

---

## IVFFlat — para datasets masivos

```sql
CREATE INDEX idx_embedding_ivfflat
ON langchain_pg_embedding
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- √N donde N = número total de vectores

SET ivfflat.probes = 10;
```

---

## HNSW vs IVFFlat

| | HNSW | IVFFlat |
|--|------|---------|
| Query latency p50 | ~2ms | ~8ms |
| Build time | Lento | Rápido |
| Memoria | Alta | Baja |
| Recall @10 | ~99% | ~95% |
| Dataset ideal | < 5M vectores | > 1M vectores |
| Añadir nuevos docs | Sin reconstruir | Reconstruir si muchos |
| Disponible desde | pgvector 0.5.0 | pgvector 0.4.0 |

---

## Operadores de distancia

```sql
<=>   -- Coseno (text embeddings — el más común)
<->   -- L2 euclidiana
<#>   -- Producto interno (negado)
```

El operador en el índice (`vector_cosine_ops`) y en el `ORDER BY` (`<=>`) DEBEN coincidir.

---

## Índice en metadata JSONB

```sql
-- Para filtrar frecuentemente por metadata
CREATE INDEX idx_metadata_gin
ON langchain_pg_embedding
USING gin (cmetadata);

-- O índice en un campo específico
CREATE INDEX idx_metadata_source
ON langchain_pg_embedding ((cmetadata->>'source'));
```

---

## Regla de oro

Inserta los datos PRIMERO, crea el índice DESPUÉS. Construir el índice sobre datos existentes es 10x más rápido que actualizarlo en cada inserción individual.

```sql
-- 1. INSERT masivo
-- 2. CREATE INDEX (construye de una vez)
-- 3. Inserciones futuras actualizan el índice automáticamente
```

Verificar que el índice se usa:
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM langchain_pg_embedding
ORDER BY embedding <=> '[...]'::vector LIMIT 4;
```
