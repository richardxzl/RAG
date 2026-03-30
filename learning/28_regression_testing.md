# Regression Testing — Detectar si un cambio empeoró el RAG

## El problema

Cambias el tamaño del chunk, ajustas el prompt, o actualizas los documentos. ¿Mejoró o empeoró el sistema? Sin un proceso de comparación, no lo sabes — y podrías mergear un cambio que degrada la calidad sin darte cuenta.

El regression testing automatiza esa comparación.

---

## Flujo

```
1. Estado actual → run_eval() → scores → guardar como baseline.json
                                                    │
                                              (commit al repo)
2. Cambias algo en el pipeline
                                                    │
3. run_eval() → scores nuevos
                     │
                     ▼
4. comparar_con_baseline() → Δ por sample
                     │
             ┌───────┴───────┐
          Δ > threshold    Δ ≤ threshold
             │                │
        REGRESIÓN           OK ✓
        → bloquear PR       → pasar
```

---

## Threshold de regresión

```python
REGRESSION_THRESHOLD = 0.05  # 5%
```

Si la faithfulness de un sample cae más de 0.05 (5 puntos porcentuales), es una regresión.

- **Muy bajo (0.02)**: demasiado sensible, falsos positivos por variabilidad del LLM
- **Recomendado (0.05)**: detecta cambios reales sin ruido
- **Muy alto (0.15)**: tolera demasiada degradación

La variabilidad natural del LLM (temperatura > 0, días diferentes) puede causar diferencias de 2-3%. El threshold debe estar por encima de ese ruido.

---

## Estructura del baseline

```json
{
  "created_at": "2026-03-30T10:00:00+00:00",
  "avg_faithfulness": 0.847,
  "avg_relevance": 0.823,
  "samples": [
    {
      "id": "q001",
      "question": "¿Qué es LCEL?",
      "faithfulness": 0.92,
      "relevance": 0.88
    }
  ]
}
```

---

## Integración en CI/CD

```yaml
# .github/workflows/eval.yml
- name: Run regression tests
  run: python 28_regression_testing.py
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

- name: Fail if regression detected
  run: |
    if [ -f "regression_detected.txt" ]; then
      echo "Regression detected!"
      exit 1
    fi
```

Para esto, `28_regression_testing.py` debe escribir `regression_detected.txt` si hay regresiones y retornar exit code 1.

---

## Guardar el baseline en el repositorio

```bash
git add eval_baseline.json
git commit -m "eval: update baseline after improving prompt"
```

El baseline es parte del código — se versiona junto con el pipeline. Cuando el cambio es intencionalmente una mejora, actualizar el baseline es parte del proceso.

---

## Cuándo regenerar el baseline

| Situación | Acción |
|-----------|--------|
| Mejora intencionalmente verificada | Actualizar baseline |
| Cambio en los documentos del corpus | Actualizar baseline |
| Cambio en el modelo LLM | Actualizar baseline |
| PR con bug fix | NO actualizar — verificar que no hay regresión |
| Ajuste de parámetros sin validar | NO actualizar — primero verificar |

---

## Limitación: variabilidad del LLM

El LLM evaluador no es determinista. Con `temperature=0` la variabilidad es mínima pero no nula. Para resultados más estables:

1. Usa `temperature=0` para el LLM evaluador
2. Promedia sobre múltiples ejecuciones (2-3 runs) antes de comparar con el baseline
3. Usa un dataset mayor (más samples = menor varianza en el promedio)
