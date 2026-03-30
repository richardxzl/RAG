# Semantic Chunking — Cortar por cambio de tema

## El problema del tamaño fijo

`RecursiveCharacterTextSplitter` divide cada N caracteres sin importar el contenido. Un párrafo sobre LangChain y el siguiente sobre fútbol pueden quedar en el mismo chunk — el LLM recibe contexto mezclado y puede confundirse.

```
Chunk 3 (tamaño fijo):
  "...Messi y Ronaldo dominaron el fútbol durante más de una
  década. Los sistemas RAG (Retrieval-Augmented Generation)
  combinan la búsqueda de información..."
                        ↑ ¡cambio de tema en medio del chunk!
```

---

## Cómo funciona SemanticChunker

1. Divide el texto en oraciones
2. Calcula el embedding de cada oración
3. Calcula la similitud coseno entre oraciones **consecutivas**
4. Cuando la similitud cae bruscamente (cambio de tema), corta ahí

```
Oración A: "LangChain es un framework..."
Oración B: "Proporciona herramientas..."       similitud: 0.89 → NO cortar
Oración C: "El fútbol es el deporte..."        similitud: 0.21 → CORTAR aquí
Oración D: "El Real Madrid y el Barcelona..."  similitud: 0.85 → NO cortar
```

---

## Implementación

```python
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings=get_embeddings(),
    breakpoint_threshold_type="percentile",  # o "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95,          # percentil 95 → solo cambios muy marcados
)

chunks = splitter.create_documents([texto])
```

Requiere `langchain-experimental`:
```bash
pip install langchain-experimental
```

---

## Tipos de breakpoint

| Tipo | Cómo funciona | Cuándo usar |
|------|--------------|-------------|
| `percentile` | Corta donde la diferencia supera el percentil N | Recomendado — independiente de escala |
| `standard_deviation` | Corta donde supera N desviaciones estándar | Distribuciones simétricas |
| `interquartile` | Basado en IQR (rango intercuartílico) | Distribuciones con outliers |

**`percentile=95`** es un buen punto de partida: solo corta donde el cambio es muy notable.

---

## Velocidad de ingesta

SemanticChunker vectoriza las oraciones para calcular las similitudes. Esto hace la ingesta aproximadamente **2× más lenta** que el chunking por tamaño.

```
RecursiveCharacterTextSplitter: ~1ms por documento
SemanticChunker:                ~50-200ms por documento (depende del modelo de embedding)
```

Para corpus grandes (>10k docs), considera si la mejora en calidad justifica el tiempo.

---

## Cuándo usar cada splitter

| Tipo de documento | Splitter recomendado |
|------------------|---------------------|
| Markdown con headers | `MarkdownHeaderTextSplitter` |
| Texto narrativo sin estructura | `SemanticChunker` |
| Código fuente | `PythonCodeTextSplitter` o `RecursiveCharacterTextSplitter` |
| HTML | `HTMLHeaderTextSplitter` |
| Documentos mixtos / genérico | `RecursiveCharacterTextSplitter` |

---

## Tradeoffs

| Ventaja | Contra |
|---------|--------|
| Chunks temáticamente coherentes | 2× más lento en ingesta |
| No requiere conocer la estructura | Tamaño de chunks variable (impredecible) |
| Funciona con texto sin formato | Requiere `langchain-experimental` |
| Reduce ruido en el retrieval | Documentos homogéneos → chunks gigantes |
