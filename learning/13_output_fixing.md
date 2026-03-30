# Output Fixing — Cuando el LLM no respeta el formato

## Por qué ocurre

El LLM genera texto estadísticamente probable, no código determinista. Aunque le des format_instructions perfectas, puede:

- Añadir texto introductorio antes del JSON ("Aquí está el análisis:")
- Omitir campos opcionales que debería incluir
- Usar tipos incorrectos (`"confianza": "alta"` en vez de `0.85`)
- Inventarse valores fuera del `Literal` definido
- Cerrar mal las llaves del JSON

Ningún parser puede evitar esto — son estrategias para **recuperarse** cuando ocurre.

---

## Estrategia 1: OutputFixingParser

Hace una segunda llamada al LLM pasando el output malo y el error de parsing para que lo corrija:

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser

base_parser = PydanticOutputParser(pydantic_object=MiModelo)

fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=llm,
)

# Si el output es malo, hace una segunda llamada automáticamente
resultado = fixing_parser.parse(output_del_llm)
```

**Internamente** construye un prompt así:
```
Instructions were given to format output in a certain way.
Here is the output the LLM gave: {output_malo}
Here is the parse error: {error}
Please try again and fix it.
```

**Cuándo usarlo**: errores sintácticos frecuentes (JSON malformado, texto extra), y el costo de una llamada extra es aceptable.

---

## Estrategia 2: RetryWithErrorOutputParser

Similar a `OutputFixingParser` pero pasa también el **prompt original** para que el LLM correctivo tenga más contexto:

```python
from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=llm,
)

# Requiere el prompt formateado además del output malo
resultado = retry_parser.parse_with_prompt(
    completion=output_malo,
    prompt_value=prompt.format_prompt(**inputs),
)
```

**Diferencia clave**: incluye el prompt original en la llamada correctiva. Útil cuando el error es semántico (el LLM no entendió qué se le pedía) y no solo sintáctico.

---

## Estrategia 3: Manejo manual con try/except

Sin llamadas extra al LLM. Predecible. Tú controlas el comportamiento en caso de fallo:

```python
from langchain_core.exceptions import OutputParserException

def parse_safe(output: str) -> MiModelo | None:
    try:
        return base_parser.parse(output)
    except (OutputParserException, ValidationError):
        return None

# Opción A: retornar objeto por defecto
def parse_with_default(output: str) -> MiModelo:
    resultado = parse_safe(output)
    if resultado is not None:
        return resultado
    return MiModelo(campo="valor_por_defecto", ...)

# Opción B: reintentar con prompt más explícito (sin OutputFixingParser)
def parse_with_retry(texto: str) -> MiModelo:
    resultado = parse_safe(chain.invoke({"texto": texto}))
    if resultado is not None:
        return resultado
    # Segunda llamada con prompt más explícito
    return base_parser.parse(chain_verbose.invoke({"texto": texto}))
```

---

## .with_structured_output() — evitar el problema de raíz

La mejor estrategia es no necesitar fixing. `with_structured_output()` usa **tool calling** nativo — el modelo está forzado a generar el schema correcto por el protocolo, no por instrucciones de texto:

```python
llm_structured = llm.with_structured_output(MiModelo)
resultado = llm_structured.invoke("Analiza este texto...")
# resultado es MiModelo directamente, sin posibilidad de formato incorrecto
```

Si el modelo soporta tool calling (Claude, GPT-4, Gemini), esta es la opción más robusta.

---

## Comparativa

| Estrategia | Llamadas LLM extra | Confiabilidad | Cuándo usar |
|-----------|-------------------|--------------|-------------|
| `OutputFixingParser` | 1 si falla | Media | Errores de formato frecuentes |
| `RetryWithErrorOutputParser` | 1 si falla | Media-Alta | Errores semánticos, necesita contexto |
| `try/except` manual | 0 | Alta (predecible) | Producción, costo controlado |
| `.with_structured_output()` | 0 | Muy alta | Modelos con tool calling |

---

## Señal de alerta

> Si `OutputFixingParser` falla frecuentemente, el problema no está en el output — está en el **prompt**. Las format_instructions no son suficientemente claras, o el modelo no es lo bastante capaz para seguirlas.

Soluciones de raíz:
1. Mejorar la descripción de los campos en el modelo Pydantic
2. Añadir ejemplos few-shot de output correcto en el prompt
3. Usar un modelo más capaz
4. Cambiar a `.with_structured_output()`
