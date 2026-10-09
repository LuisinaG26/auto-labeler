# Etiquetador reutilizable de object detection: plan de diseño (v3)

Objetivo principal: **reducir el esfuerzo humano de etiquetado**.
Objetivo de diseño: **reutilizable**. Un proyecto nuevo se define con una carpeta y un YAML, sin tocar código.

## Estado de avance

| Fase | Estado |
|---|---|
| 0. Detector en MCP y evaluación | **Completada.** Resultados en la sección 8.1. |
| 1. Filtro de calidad | **Completada.** Resultados en la sección 8.2. |
| 2. Etiquetado por lotes | **Completada.** Resultados en la sección 8.3. |
| 3. Verificación y curador | **Completada con partes diferidas.** Resultados en la sección 8.4. Quedan para la fase 4 la revisión de objetos faltantes y la similitud por embeddings. |
| 4. Bucle etiquetado y verificación | **Implementada y medida, con un criterio no cumplido.** El flujo corre de punta a punta y es reanudable, pero **el bucle no mejoró el resultado** con las estrategias probadas, y casi nada se acepta solo a nivel de imagen. Ver 8.5. |
| 5 a 7 | Pendientes |

Hecho hasta ahora: imágenes Docker (GPU y liviana) verificadas, servidores MCP del detector y del filtro de calidad, cargador de imágenes con orientación EXIF, esquema de `project.yaml`, set dorado de 100 imágenes con las etiquetas originales de `laborotomato`, métricas y evaluación del detector, filtro de calidad calibrado sobre `tomato/`, etiquetado por lotes con exportación a YOLO, carga de preanotaciones en Label Studio, modelo de visión local con Qwen3-VL, verificador de cajas con confianza calibrada, curador de imágenes dudosas, las tres correcciones de genericidad, soporte de mosaico en el detector, el grafo de LangGraph con el bucle y su comando `run`, un segundo proyecto (`tomatod`) con su set dorado y mediciones del detector, y 149 pruebas automáticas.

Historial de cambios: la v2 eliminó Strands por redundancia con LangGraph y ajustó los modelos a 8 GB de VRAM. La v3 refleja el código existente y los resultados de la fase 0.

## 1. Técnica elegida: LangGraph + MCP

| Capa | Tecnología | Por qué |
|---|---|---|
| Orquestación | **LangGraph** | El flujo es un grafo con ramas (descartar, etiquetar, revisar), estado persistente para reanudar lotes y pausa para el humano. |
| Herramientas | **MCP** | Detector, calidad y dataset son servidores independientes. Cualquier proyecto o cliente MCP los reutiliza. |

**Por qué LangGraph y no Strands:** el pipeline es casi determinista y trabaja por lotes. Eso pide checkpoints y rutas explícitas, que LangGraph da de fábrica. Con modelos locales chicos, además, conviene que los agentes devuelvan JSON con esquema y no que decidan solos qué herramienta llamar, porque el tool-calling de un modelo de 4B es poco fiable.

**Portabilidad:** la lógica de cada agente vive en funciones Python puras (entrada y salida tipadas). LangGraph solo las conecta. Si más adelante querés Strands u otro framework, se cambia el conectado sin tocar agentes ni servidores MCP.

## 2. Agentes y flujo: filtrar, etiquetar, verificar en bucle, revisar

### Flujo de ejecución

```
Filtro de calidad ──► Etiquetado automático ──► Verificación ──► Revisión manual
 (apartar imágenes)     (detector)                 │   ▲          (solo casos puntuales)
                                                   │   │
                                                   ▼   │
                                            ¿se cumple la condición?
                                              no: reintentar con otra
                                                  estrategia ───────┘
```

| # | Etapa | Modelo | Función |
|---|---|---|---|
| 1 | **Filtro de calidad** | Código y VLM | El prefiltro mide desenfoque, exposición, resolución y duplicados. El **curador** (VLM) juzga si las clases se distinguen. Cada imagen queda `keep`, `drop` o `unsure`, con motivo. Lo apartado no llega al detector. |
| 2 | **Etiquetado automático** | Detector | Llama al detector por MCP con la estrategia actual, aplica NMS y umbrales por clase. |
| 3 | **Verificación** | VLM, embeddings | Valida cada caja (clase correcta, no es falso positivo) y revisa si faltan objetos en la imagen. Calcula la confianza final. |
| 4 | **Decisión** | Código | Con la confianza decide: aceptar, reintentar con otra estrategia o pasar a revisión. |
| 5 | **Revisión manual** | Humano | Label Studio, solo los casos que no se resolvieron solos, con el motivo a la vista. |

Los agentes son nodos con salida estructurada. No hay bucle libre de herramientas: el único ciclo es el de la etapa 4, y está acotado.

### Bucle de verificación (etiquetado ↔ verificación)
Si la verificación no queda conforme, la imagen vuelve al etiquetado con la **siguiente estrategia** de una lista ordenada. Cada estrategia combina modelo, umbral y preproceso. Ejemplo:

| Intento | Estrategia |
|---|---|
| 1 | Grounding DINO tiny, umbral 0,2 |
| 2 | Grounding DINO tiny con mosaico y ampliación, para objetos chicos |
| 3 | Grounding DINO base |
| 4 | Otro backend (SAM 3, Florence-2 o Qwen3-VL con localización) |

Reglas para que el bucle sea seguro y barato:
- **Solo se reintenta lo que falló.** Las imágenes ya aceptadas no vuelven a pasar. En cada vuelta se agrupan por estrategia para cargar cada modelo una sola vez por la VRAM.
- **Se acumula, no se reemplaza.** Los resultados de varios intentos se fusionan. Si dos modelos distintos coinciden en una caja, su confianza sube: el acuerdo entre modelos es una señal.
- **Condiciones de parada, por proyecto en el YAML:**
  1. Todas las cajas de la imagen superan el umbral de aceptación y la verificación no ve objetos faltantes.
  2. Se llegó al máximo de intentos.
  3. Un intento no mejoró la confianza por encima de un margen mínimo.
  4. Se agotó el presupuesto de tiempo.
- **Al agotarse sin éxito**, la imagen pasa a revisión manual con el mejor resultado obtenido y el motivo.
- **Cada iteración guarda checkpoint** y queda registrada qué estrategia produjo cada caja.

Riesgo propio del bucle: si el mismo modelo etiqueta y verifica, itera hasta que el juez se convence de sus propios errores. Mitigaciones: el verificador usa un modelo distinto del etiquetador, la precisión de lo autoaceptado se mide contra el set dorado y se audita una muestra en `tomato/`.

### Orden de ejecución: por etapas, no imagen por imagen
Con 8 GB de VRAM no caben a la vez el VLM y el detector con holgura. Si cada imagen recorriera todo el pipeline, los modelos se cargarían y descargarían constantemente. Por eso el grafo procesa **lotes por etapa**: primero todo el lote por el filtro, luego por el detector, luego por el verificador, y las que fallan vuelven al detector en grupo. Cada etapa guarda un checkpoint, así que se puede interrumpir y reanudar.

### Confianza combinada
La confianza no es solo el score del detector. Se combina:
1. Score del detector.
2. Probabilidad del VLM sobre el recorte.
3. Acuerdo entre dos detectores (Grounding DINO y SAM 3), cuando esté el segundo.
4. Estabilidad de la caja ante flip y cambio de escala.
5. Similitud con las imágenes de referencia de la clase, si el humano las aportó (ver 3.1).

Los umbrales se calibran con un set dorado para que "alta confianza" signifique que casi siempre acierta.

## 3. Hardware y modelos (esta notebook)

Verificado en esta máquina:

| Recurso | Valor |
|---|---|
| GPU | RTX 4060 Laptop, 8 GB de VRAM (1 GB ya en uso por el escritorio) |
| RAM | 32 GB |
| Docker | 29.8, WSL2 |
| Docker con GPU | **Funciona**: un contenedor CUDA vio la GPU |
| Python | 3.11.9 |
| Ollama | No instalado (irá en contenedor) |

### Elección
| Rol | Modelo | VRAM aprox. | Licencia | Comentario |
|---|---|---|---|---|
| Detector (MVP) | **Grounding DINO**, variante tiny o base | menos de 1 GB en fp16 | Apache 2.0 | Referencia de detección guiada por texto. |
| Detector (fase posterior) | **SAM 3** | 3 a 4 GB | Licencia propia de Meta | Mejor resultado en el benchmark que encontré. Pesos probablemente con acceso restringido en Hugging Face (a verificar). |
| Detector alternativo | **Florence-2** (grounding de frases) y **Qwen3-VL** (localización por texto) | 1 a 4 GB | MIT y abierta | Candidatos a comparar en la fase 0 y a usar como segunda opinión. |
| VLM | **Qwen3-VL 4B, variante `instruct`** en Ollama | unos 3 a 4 GB | Abierta | Entra con holgura. **La variante de razonamiento (`qwen3-vl:4b`) no sirve para este uso**: emite un bloque de pensamiento antes de responder y no permite leer probabilidades. El de 8B queda justo y solo se probaría si el 4B no alcanza. |
| Calidad | OpenCV + hash perceptual | CPU | Libre | Sin modelo. |

| Embeddings de referencia | **DINOv2 small** o CLIP | menos de 1 GB | Abierta | Solo se carga si el proyecto tiene referencias. |

Descartado: YOLO-World (GPL-3.0, menos preciso). OWLv2 rindió peor solo con texto, pero queda como backend opcional por su detección guiada por imagen.
Los benchmarks salen de un solo agregador, así que se validan con tus datos en la fase 0.

**Resultado medido en la fase 0:** Grounding DINO tiny queda como modelo por defecto. La variante base mejora poco al localizar, rinde peor por clase y tarda cerca de 30% más (detalle en la sección 8.1). Los demás backends siguen pendientes de comparar.

### Restricción: no se entrena ningún detector
El detector debe entender las clases por texto. Solo se aceptan modelos guiados por prompt: tipo Grounding DINO o un modelo de visión-lenguaje con localización. No hay fine-tuning ni entrenamiento posterior.

Consecuencias de diseño:
- **El detector es intercambiable.** El servidor `mcp-detector` tiene backends elegibles por YAML: `grounding_dino`, `sam3`, `owlv2`, `florence2` y `qwen3_vl` (localización por texto). Todos reciben el mismo contrato: imagen, lista de clases con prompts, umbrales. Devuelven cajas, clase y score.
- **Se evalúan todos en la fase 0** sobre `tomatod` y se elige por precisión y recall. Los modelos de visión-lenguaje generales suelen localizar peor, pero entienden mejor descripciones largas, así que sirven como segundo detector o como verificador.
- **La calidad se mejora sin entrenar**, con estas palancas por proyecto:
  1. Ajustar `prompt` y `synonyms` de cada clase.
  2. Detectar por mosaico (tiling) y escalado para objetos chicos.
  3. Combinar dos detectores y quedarse con las cajas donde coinciden.
  4. Verificar cada recorte con el VLM y descartar los falsos positivos.
  5. Guardar los prompts que mejor funcionan en el YAML del proyecto.
- **La reducción de esfuerzo viene del router**, no de un modelo propio: lo que hay acuerdo y alta confianza se acepta solo, y el humano revisa el resto.

### 3.1 Imágenes de referencia por clase (opcional)
El humano puede dar unas pocas imágenes de ejemplo de cada clase, de 2 a 5 por clase. Si no las da, el flujo funciona igual solo con texto.

Formato: una carpeta por clase dentro del proyecto, con recortes del objeto. Opcionalmente, una carpeta de negativos con cosas que se parecen y no son de la clase.

```
projects/<proyecto>/refs/<clase>/ok_1.png
projects/<proyecto>/refs/<clase>/ok_2.png
projects/<proyecto>/refs/<clase>/not_1.png   # negativo, opcional
```

Dónde se usan las referencias:

| Componente | Cómo las usa |
|---|---|
| Curador | Las recibe como imágenes de ejemplo en el prompt del VLM para juzgar mejor qué es "visible con claridad". |
| Verificador | Muestra la referencia junto al recorte y pregunta si son el mismo tipo de objeto. Es la mejora más directa sobre el texto solo. |
| Similitud de embeddings | Se compara el recorte con las referencias usando un modelo pequeño de embeddings (DINOv2 small o CLIP, menos de 1 GB). No usa LLM y suma una señal barata a la confianza. |
| Detector | **Depende del backend.** Grounding DINO solo entiende texto y las ignora. SAM 3 y OWLv2 aceptan ejemplos visuales. Por eso OWLv2 vuelve como backend opcional para detección guiada por imagen, aunque rindió peor solo con texto. |

Límites a tener en cuenta:
- Con 8 GB de VRAM se pasan como máximo 2 o 3 referencias por clase al VLM, para no agotar contexto y memoria.
- Las referencias son datos del proyecto, no del código. Respetan la regla de la sección 6.
- La fase 0 mide el efecto: se corre cada configuración con y sin referencias sobre el set dorado y se reporta la diferencia.

## 4. Docker

**Construido y verificado.** Hay dos imágenes y un `docker-compose.yml` con perfiles.

| Imagen | Tamaño | Contenido |
|---|---|---|
| `autolabel-gpu` | 9,7 GB | torch 2.6 con CUDA 12.4, transformers 5.19, timm, OpenCV, MCP. Para los servidores con modelos. |
| `autolabel-light` | unos 1,2 GB | LangGraph, adaptadores MCP, cliente OpenAI, OpenCV, imagehash, pycocotools, pytest y ruff. Para el orquestador, calidad y dataset. |

| Servicio del compose | Estado |
|---|---|
| `gpu`, `light` (perfil `tools`) | Para correr comandos puntuales dentro de cada imagen |
| `mcp-detector` (perfil `services`) | Implementado. Usa transporte `streamable-http` en el puerto 8001 |
| `mcp-quality` (perfil `services`) | Implementado. Usa la imagen liviana y el puerto 8002 |
| `ollama` (perfil `vlm`) | Implementado, con `qwen3-vl:4b-instruct`. La imagen oficial es grande (unos 9 GB) y cada modelo ocupa unos 3,3 GB en su volumen |
| `label-studio` (perfil `review`) | Implementado con la imagen oficial 1.22.0, solo en `127.0.0.1:8080`. Sirve las imágenes locales desde `/data` en solo lectura |
| `mcp-dataset`, `orchestrator` | Pendientes |

Detalles prácticos:
- El código y la carpeta `projects/` se montan como volumen, y el dataset va en solo lectura. Las imágenes no contienen datos ni código del proyecto.
- Los pesos de Hugging Face se guardan en un volumen de caché.
- MCP queda fijado a la serie 1.x en ambas imágenes, porque la versión 2 instalada por defecto en la imagen GPU no era compatible con los adaptadores de LangGraph.
- Las versiones exactas están en `docker/requirements-*.lock.txt`.
- Disco: hay unos 350 GB libres, suficiente.

## 5. Datos de prueba: `dataset-tomates/`

| Carpeta | Contenido | Uso |
|---|---|---|
| `tomato/` | 895 PNG de 400x500, **sin etiquetas** | Proyecto de prueba principal (se usa un subconjunto, ver abajo) |
| `laborotomato-DatasetNinja/` | 804 fotos de celular (3120x4160 y similares) con **polígonos** y 6 clases: `b_` y `l_` por cada uno de `green`, `half_ripened`, `fully_ripened` | **Set dorado.** Al unir `b_` y `l_` quedan exactamente tus 3 clases: verde, semimaduro, maduro. Los polígonos se convierten a cajas. |
| `tomatod-DatasetNinja/` | 277 imágenes de 2000x2000 (5 de 1000x1000) con **cajas** de 3 clases: `unripe`, `semi-ripe`, `fully-ripe` | **Segundo proyecto** para la prueba de genericidad, y benchmark de la fase 0 |
| `Rob2Pheno...` | 123 TIFF más JSON | Fuera del alcance inicial |

**Por qué `laborotomato` es el set dorado:** es el que mejor coincide con tus clases y con el tipo de foto de `tomato/` (primer plano, tomada a mano, invernadero). `tomatod` son escenas amplias con frutos muy chicos, tomadas por un robot, así que se parece menos. Su ventaja es que ya trae 3 clases, con otros nombres, lo que lo vuelve ideal como segundo proyecto.

Conteo de cajas: `laborotomato` trae unas 10.600 instancias y `tomatod` unas 2.800. En ambos domina la clase verde. Hay que tenerlo en cuenta porque la maduración total y la semimadura tienen menos ejemplos.

**Licencia:** ambos datasets son CC BY-NC-SA 4.0, uso no comercial. Sirven para estas pruebas, pero no para un producto comercial.

**Dato a verificar:** supongo que `b_` y `l_` distinguen tomate grande de cherry, pero los README no lo aclaran. No afecta al plan porque se unen en una sola clase.

Lo que vi en `tomato/`: imágenes de invernadero con desenfoque real, hojas y tallos que tapan los frutos, tomates cherry en racimos y objetos muy chicos. Es un buen caso para el curador (hay imágenes casi inservibles) y exigente para el detector.

### Tamaño de la prueba (hardware limitado, objetivo: todo el flujo hoy)

| Conjunto | Cantidad | Para qué |
|---|---|---|
| Set dorado: subconjunto de `laborotomato` | **100 imágenes** con semilla fija, **reducidas a unos 400x500** | Calibrar umbrales y medir, ya con etiquetas |
| Referencias | 3 recortes por clase, tomados de otras imágenes de `laborotomato` que no estén en el set dorado | Probar la sección 3.1 sin fuga de datos |
| Subconjunto de `tomato/` | **120 imágenes** con semilla fija | Corrida completa del flujo sobre datos sin etiquetas, incluida la revisión humana |
| Segundo proyecto: `tomatod` | **40 imágenes**, también reducidas | Probar que el flujo es genérico |

Por qué estos números: 100 imágenes de `laborotomato` suman cientos de cajas por clase, suficiente para calibrar sin etiquetar nada a mano. 120 de `tomato/` alcanzan para ver las rutas del grafo y el volumen de revisión. Más imágenes alargan las corridas y no cambian lo que se aprende hoy.

Las imágenes del set dorado se reducen a un tamaño parecido al de `tomato/` porque un modelo que ve un tomate de 400 píxeles de ancho no rinde igual que con uno de 4000. Esa reducción es parte del preproceso del subconjunto y no toca el original.

Cómputo medido en esta GPU: el detector tarda unos 300 ms por imagen al buscar una sola clase y unos 800 ms con tres prompts. El tiempo del VLM de 4B todavía no está medido. Como no hay que etiquetar a mano, el tiempo del día se va en el cómputo y en revisar.

**Orientación EXIF (hallazgo de la fase 0):** 70 de las 161 imágenes del conjunto Test de `laborotomato` están guardadas en horizontal con una marca EXIF de rotación, y sus etiquetas están en el marco ya girado. Si no se aplica la rotación, las imágenes quedan deformadas y las cajas desalineadas. Las etiquetas del dataset se usan sin modificar y lo que se adapta es la imagen. Todo el flujo debe cargar imágenes con `autolabel/imaging.py`, y el armado del set dorado falla si las proporciones no coinciden. Las 100 imágenes del set dorado ya se guardan giradas y sin marca EXIF.

**Límite que hay que decir claro:** el set dorado mide el flujo en el dominio de `laborotomato`, no en `tomato/`. Las fotos de `tomato/` son más borrosas y tienen frutos más chicos, así que los números reales ahí pueden ser peores. Para tener una estimación sin etiquetar desde cero, la revisión humana de `tomato/` sirve de auditoría: se cuenta cuántas cajas autoaceptadas el humano corrige al revisar una muestra de unas 15 imágenes.

La carpeta del subconjunto se genera con un comando y una semilla, sin mover ni modificar `dataset-tomates/`. La API key (Gemini u OpenRouter) queda fuera de hoy. El modelo de visión ya queda detrás de una interfaz con proveedor configurable, así que agregarla más adelante es un cambio de configuración.

### Estrategia de evaluación
1. **Fase 0:** correr cada detector sobre el set dorado de `laborotomato` y medir precisión y recall por clase. Es automático: las etiquetas ya existen.
2. **Dos formas de resolver la maduración, a comparar con el mismo set dorado:**
   - **Directa:** tres clases con su propio prompt (por ejemplo "green unripe tomato", "half-ripe orange tomato", "ripe red tomato"). Simple, pero los detectores de texto distinguen mal los matices de color.
   - **En dos pasos:** detectar "tomato" y luego clasificar cada recorte como verde, semimaduro o maduro con el VLM y las referencias. Suele ser más fiable para los atributos de color.
   El plan soporta las dos por configuración, y el resultado decide cuál queda por defecto.
3. **Calibración:** los umbrales por clase se eligen con el set dorado.
4. **Auditoría en `tomato/`:** la revisión humana mide cuánto se equivoca el flujo en el dominio real.

## 6. Genericidad: el flujo no sabe de qué es el proyecto

Regla de oro: **todo lo que cambia entre proyectos vive en la carpeta del proyecto. Nada de eso vive en el código, ni en los prompts de los agentes, ni en los servidores MCP.**

### 6.1 Contrato del proyecto
Un proyecto es una carpeta con un `project.yaml` y sus imágenes. Se valida con un esquema al arrancar y falla con un mensaje claro si falta algo.

```yaml
name: tomates
images: ./tomato
scene: "fotos de cultivo de tomate en invernadero, tomadas a mano"   # contexto para el curador
classes:
  - name: verde
    prompt: "green unripe tomato"           # texto que recibe el detector
    synonyms: ["unripe tomato"]
    definition: "tomate completamente verde, visible al menos en 50%"
    not_this: ["hoja", "pimiento verde"]    # negativos para el verificador
    source_labels: [b_green, l_green]       # etiquetas del dataset de origen, solo para evaluar o importar
    references:                             # opcional, 2 a 5 recortes por clase
      positive: [./refs/verde/ok_1.png, ./refs/verde/ok_2.png]
      negative: [./refs/verde/not_1.png]
    min_box_px: 12
    thresholds: { auto_accept: 0.85, review: 0.40 }
  - name: semimaduro
    prompt: "half-ripe tomato"
    definition: "tomate con transición de verde a rojo o naranja"
    source_labels: [b_half_ripened, l_half_ripened]
    # ... mismos campos
  - name: maduro
    prompt: "ripe red tomato"
    definition: "tomate rojo o rosado completamente maduro"
    source_labels: [b_fully_ripened, l_fully_ripened]
    # ... mismos campos
detector:
  backend: grounding_dino
  model_id: IDEA-Research/grounding-dino-tiny
  box_threshold: 0.25
  agnostic_prompt: "tomato"        # agrupa todas las clases, para localizar sin clasificar
quality:                           # sin blur_min/blur_ok no se filtra por desenfoque: se calibran con
  blur_min: 25                     # `quality --calibrate` y se confirman mirando muestras
  blur_ok: 35
curator: { keep_if: "al menos una clase objetivo claramente visible" }
labeling:                          # bucle etiquetado ↔ verificación
  mode: agnostic
  strategies:                      # se prueban en orden hasta cumplir la condición
    - { model_id: IDEA-Research/grounding-dino-tiny, box_threshold: 0.2 }
    - { model_id: IDEA-Research/grounding-dino-base, box_threshold: 0.15 }
  max_iterations: 3
  min_gain: 0.02                   # si un intento no mejora lo resuelto en este margen, se corta
  time_budget_min: 30
export: [yolo, coco]
```

El esquema está implementado en `autolabel/config.py`, incluidos los bloques `quality`, `vlm` y `labeling`, y es estricto: una clave desconocida, una clase repetida o una etiqueta de origen en dos clases hacen fallar la carga con un mensaje claro. La carpeta del proyecto no se escribe en el YAML, la calcula el cargador. Los umbrales por clase (`thresholds`) ya existen en el esquema y los usa el router del bucle.

Otro proyecto cambia solo este archivo. Por ejemplo, uno de vehículos tendría `car`, `truck`, `bus` y su propia definición de "imagen útil".

### 6.2 Cómo se evita que algo quede fijo en el código
| Punto de fuga | Cómo se resuelve |
|---|---|
| Nombres de clase en prompts de agentes | Los prompts son **plantillas** con variables (`{scene}`, `{classes}`, `{definition}`, `{not_this}`). Se rellenan desde el YAML. |
| Salida del VLM con clases fijas | El esquema JSON de salida se **construye en tiempo de ejecución** con las clases del proyecto como valores permitidos. |
| Servidores MCP con estado del proyecto | Son **sin estado**. Cada llamada recibe clases, prompts y umbrales como argumentos. Un mismo servidor atiende a cualquier proyecto. |
| Umbrales globales | Son por clase y por proyecto. Se calibran con el set dorado de cada proyecto. |
| Criterio de "imagen útil" | Texto libre en `curator.keep_if` más las métricas de calidad del YAML. |
| Estado y caché mezclados | Checkpoints, caché y resultados viven dentro de la carpeta del proyecto, así dos proyectos no se pisan. |
| Formato de entrada y salida | Importadores y exportadores intercambiables (YOLO, COCO, y conversión de polígonos a cajas). |
| Clases con atributos (por ejemplo madurez) | Campo opcional `attributes` por clase. Si no existe, el paso no corre. |
| Imágenes de referencia | Opcionales y por clase. Si una clase no las tiene, los pasos que las usan se saltan y el flujo sigue solo con texto. |

### 6.3 Cómo se comprueba que es genérico
Esto pasa a ser un criterio de aceptación, no una intención:
1. **Prueba de texto fijo:** una verificación automática falla si aparece una palabra específica de un proyecto (por ejemplo "tomato") en `autolabel/`, `services/` o en las plantillas. **Ya implementada** en `tests/test_generic.py` y pasa.
2. **Segundo proyecto obligatorio:** antes de dar por buena la fase 3, el mismo flujo se corre sobre otro proyecto con clases distintas, **solo creando una carpeta nueva**. Candidato inmediato: `tomatod`, con sus 3 clases con otros nombres (`unripe`, `semi-ripe`, `fully-ripe`) y otro tipo de imagen. Solo cambia su `project.yaml`. Más adelante conviene sumar uno que no sea tomate para probar de verdad que no hay arrastre.
3. **Andamiaje:** `autolabel init <carpeta>` genera un `project.yaml` de plantilla, para que crear un proyecto nuevo no exija copiar y editar a mano.

### 6.4 Lo que sí puede limitar la genericidad
- Un detector open-vocabulary no conoce clases muy específicas o de dominios raros (piezas industriales, patologías). Para esos casos el flujo se mantiene, pero la calidad depende de los prompts, los sinónimos, las referencias y la verificación con el VLM. Como no se entrena, en esos dominios el humano revisará una parte mayor.
- La calidad del VLM de 4B para juzgar crece con definiciones claras en `definition` y `not_this`. Un YAML vago da etiquetas peores.

## 7. Estructura del repositorio

Lo marcado con (hecho) existe; el resto es lo planificado.

```
AI Day/
  PLAN.md
  docker-compose.yml                (hecho)
  pyproject.toml                    (hecho: ruff y pytest)
  .dockerignore                     (hecho)
  docker/                           (hecho: dos Dockerfiles, requirements y lock)
  autolabel/
    config.py                       (hecho: esquema de project.yaml)
    geometry.py                     (hecho: cajas e IoU con NumPy)
    imaging.py                      (hecho: carga con orientación EXIF)
    quality.py                      (hecho: nitidez, exposición, duplicados, veredicto)
    reporting.py                    (hecho: hojas de contacto para revisar a ojo)
    datasets/ninja.py               (hecho: lector DatasetNinja)
    detectors/                      (hecho: contrato, registro y Grounding DINO)
    graph.py                        (hecho: LangGraph con el bucle, checkpoints en SQLite)
    loop.py                         (hecho: estado de cada imagen, fusión de cajas y decisión)
    tiling.py                       (hecho: mosaico de tiles para objetos chicos)
    labeling.py                     (hecho: selección de imágenes, lote reanudable)
    vlm/                            (hecho: interfaz común y adaptador de Ollama)
    prompts/                        (hecho: plantillas con variables, sin nombres de clase)
    verification.py                 (hecho: recorte, control de objeto único, clase)
    verifying.py                    (hecho: verificación por lotes reanudable)
    calibration.py                  (hecho: confianza combinada con regresión logística)
    curation.py                     (hecho: curador de imágenes dudosas)
    references.py                   (hecho: referencias por clase y hoja para el modelo)
    detector_client.py              (hecho: cliente MCP del detector y modo local)
    export.py                       (hecho: exportación YOLO)
    label_studio.py                 (hecho: tareas con preanotaciones y cliente de la API)
    scaffold.py                     (hecho: plantilla de un proyecto nuevo)
    cli.py                          (hecho: init, quality, label, curate, verify, run, labelstudio)
  services/
    _common.py                      (hecho: rutas permitidas)
    mcp_detector/  mcp_quality/     (hecho)
    mcp_dataset/
  eval/                             (hecho: set dorado y su partición, referencias, métricas, evaluación del
                                     detector, del filtro, del verificador y del flujo completo, prueba MCP)
  tests/                            (hecho: 156 pruebas)
  projects/tomates-gold/            (hecho: project.yaml, imágenes, gold.json, out/)
  projects/tomates/                 (hecho: project.yaml sobre tomato/, con calibración de calidad)
  projects/tomatod/                 (hecho: segundo proyecto, otras clases, solo un project.yaml nuevo)
  dataset-tomates/                  datos originales, solo lectura
```

## 8. Fases

| Fase | Entregable | Criterio de salida |
|---|---|---|
| 0 (**hecha**) | Servidor `mcp-detector` con Grounding DINO en Docker, conversión de `laborotomato` a cajas y evaluación sobre el set dorado | Tenemos precisión y recall de referencia. Cumplido, ver 8.1 |
| 1 (**hecha**) | **Filtro de calidad:** `mcp-quality` con desenfoque, exposición, resolución y duplicados. Lo apartado se registra con su motivo y no se borra | Aparta las imágenes borrosas de `tomato/` sin tocar las buenas. Validado con desenfoque artificial sobre el set dorado y con el corte confirmado a ojo |
| 2 (**hecha**) | **Etiquetado por lotes:** CLI que corre el detector solo sobre las imágenes que pasaron el filtro y exporta YOLO. Importación a Label Studio | Cajas visibles sobre las imágenes filtradas |
| 3 (**hecha, con partes diferidas**) | **Verificación con Qwen3-VL 4B:** valida cada caja, detecta cajas de grupo, clasifica la madurez en dos pasos sobre cada recorte y usa imágenes de referencia. Incluye el **curador** que decide `keep`, `drop` o `unsure`. **Diferido a la fase 4:** revisión de objetos faltantes por imagen y similitud por embeddings | Confianza combinada calibrada con el set dorado, y diferencia medida con y sin referencias. Cumplido, ver 8.4 |
| 4 (**hecha, criterio de mejora no cumplido**) | **Grafo LangGraph con el bucle etiquetado ↔ verificación:** lista de estrategias, condiciones de parada, checkpoints por lote y registro de qué estrategia produjo cada caja | Lote reanudable (**cumplido**). El bucle mejora la autoaceptación sobre el intento único (**no cumplido**: las rondas 2 y 3 sumaron 2 objetos correctos). **El mismo flujo corre en un segundo proyecto con otras clases, solo con una carpeta nueva** (**cumplido con `tomatod`**) |
| 5 (**carga hecha, revisión humana pendiente**) | **Revisión manual:** Label Studio recibe solo los casos no resueltos, con el motivo a la vista. La carga se probó con datos reales de `tomatod` y queda en un proyecto aparte | Humano revisa un subconjunto y se mide el tiempo |
| 6 | Más backends en la lista de estrategias: SAM 3, Florence-2, OWLv2 y Qwen3-VL con localización | Mejora medida frente a Grounding DINO solo |
| 7 | Ajuste de prompts, mosaico y ampliación, medido sobre el set dorado | Sube la autoaceptación sin bajar la precisión |

**Para hoy:** se recorren las fases 1 a 5 sobre el subconjunto de 120 imágenes de `tomato/`, con una versión mínima de cada una. Las fases 6 y 7 quedan para después si el tiempo no alcanza.

La fase 0 va primero porque el detector es la pieza que más pesa en la calidad. Si falla en tomates chicos o tapados, es mejor saberlo antes de construir el resto.

### 8.1 Resultados de la fase 0

Set dorado: 100 imágenes de `laborotomato` reducidas a 500 px de lado mayor, con 1.800 cajas: 1.139 verdes, 330 semimaduras y 331 maduras. Las 1.678 que miden 10 px o más son las que cuentan en las métricas, y las menores se ignoran. IoU 0,5.

| Modo | Modelo | AP50 | Mejor F1 | Precisión | Recall | Tiempo por imagen |
|---|---|---|---|---|---|---|
| Localizar "tomato" | tiny | 0,548 | 0,634 | 0,75 | 0,55 | 300 ms |
| Localizar "tomato" | base | 0,566 | 0,657 | 0,76 | 0,58 | 390 ms |
| Un prompt por clase, promedio | tiny | 0,264 | | | | 810 ms |
| Un prompt por clase, promedio | base | 0,211 | | | | 1.090 ms |

AP50 por clase con tiny y un prompt por clase: verde 0,37, maduro 0,25, semimaduro 0,17.

Conclusiones:
- **Localizar frutos funciona** sin entrenar: encuentra algo más de la mitad con 75% de precisión. El umbral óptimo es bajo, de 0,18 a 0,21, así que el umbral por defecto de 0,25 pierde recall.
- **Clasificar madurez con el prompt no funciona.** Confirma que la madurez debe resolverse en dos pasos, con el VLM y las referencias sobre cada recorte (fase 4).
- **El detector deja la mitad de los frutos sin encontrar.** Eso es un techo para el recall del flujo completo. Las palancas de la fase 7 (mosaico, escalado, segundo detector) apuntan a esto.
- **Tiny por defecto.** Base casi no mejora al localizar y rinde peor por clase.

Estos números miden el dominio de `laborotomato`. En `tomato/` pueden ser peores (ver riesgos).

### 8.2 Resultados de la fase 1: filtro de calidad

**Qué mide.** Cada imagen se evalúa sin modelos: resolución mínima, exposición (brillo medio y fracción de píxeles negros o quemados), nitidez y duplicados. El veredicto es `keep`, `drop` o `unsure` con el motivo, y se guarda en `out/quality.json` dentro del proyecto. No se borra ni se mueve nada del dataset original. Se usa con `python -m autolabel.cli quality --project <carpeta>` y también como herramienta MCP (`assess_images`).

**Cambio de medida durante la fase.** La primera versión usaba la varianza global del Laplaciano. Castiga las escenas con superficies lisas, como una hoja grande y nítida. Se reemplazó por el percentil 90 de la varianza por zonas en una grilla de 4x4. Sobre imágenes reales de `tomato/` donde las dos medidas discrepaban, la nueva coincidió mejor con lo que se ve.

**Calibración sobre `tomato/`.** El `tomato/` típico puntúa unas 16 veces menos que el set dorado, así que los cortes no pueden salir del set dorado. Se fijaron mirando imágenes a lo largo de la distribución: por debajo de 25 son visiblemente borrosas, de 25 a 35 blandas pero utilizables, y desde 35 nítidas. La escala depende de la cámara y hay que recalibrar en cada proyecto.

| Veredicto sobre las 895 imágenes de `tomato/` | Cantidad |
|---|---|
| `keep` | 702 |
| `unsure` (desenfoque leve) | 105 |
| `drop` por desenfoque | 87 |
| `drop` por sobreexposición | 1 |

**Validación con desenfoque artificial.** Se aplicó desenfoque gaussiano creciente al set dorado y se midió el detector sobre las imágenes degradadas:

| Desenfoque | Nitidez | Recall del detector | F1 |
|---|---|---|---|
| ninguno | 989 | 0,56 | 0,63 |
| leve | 100 | 0,52 | 0,62 |
| medio | 16 | 0,41 | 0,54 |
| fuerte | 3 | 0,24 | 0,37 |

La nitidez cae de forma monótona y el recall cae con ella. Los cortes de 25 y 35 caen entre el desenfoque leve y el medio: se conservan las imágenes que el detector todavía maneja y se apartan las que pierden casi la mitad del recall.

**Duplicados.** Con el hash perceptual, el par más cercano de `tomato/` está a distancia 8 y son cuadros consecutivos de la misma escena. A distancia 12 ya son escenas distintas. Se fijó el corte en 8 para este proyecto. Con ese corte no se marcó ninguna imagen, porque el único par claro tiene una imagen ya apartada por borrosa. El valor por defecto del esquema es 4, más estricto.

**Limitaciones**
- La nitidez mide enfoque, no utilidad. Unos pocos tomates grandes de superficie lisa quedaron en `drop` aunque pueden ser usables. Por eso lo apartado se registra con el motivo y se puede recuperar.
- La zona `unsure` (105 imágenes) queda para el curador con modelo de visión de la fase 3, que juzga si las clases se distinguen.
- El corte se confirmó mirando muestras, no con una verdad etiquetada de qué imágenes son inservibles.

### 8.3 Resultados de la fase 2: etiquetado por lotes

**Qué hace.** `python -m autolabel.cli label --project <carpeta>` toma solo las imágenes que el filtro dejó como `keep` (o también `unsure` con `--include-unsure`), llama al detector por MCP y guarda todo con sus scores en `out/labels.json`. Exporta a YOLO en `out/yolo/` y genera una vista previa con las cajas dibujadas. Si se interrumpe, al repetirlo retoma solo lo pendiente, siempre que la configuración sea la misma. `--limit` y `--seed` toman una muestra reproducible. El modo `agnostic` busca todo con un prompt, y el modo `direct` usa un prompt por clase. Falla con un mensaje claro si no se corrió antes el filtro de calidad.

**Corrida sobre `tomato/`**, 120 imágenes filtradas con semilla 0, umbral 0,2, por MCP:

| Medida | Valor |
|---|---|
| Cajas | 597, unas 5 por imagen |
| Imágenes sin ninguna caja | 1 |
| Score de las cajas, percentiles 10 / 50 / 90 | 0,22 / 0,32 / 0,55 |
| Velocidad | unas 2,5 imágenes por segundo |

**Lo que se ve en la vista previa**
- Los tomates claros reciben cajas ajustadas con scores de 0,4 a 0,65.
- Los falsos positivos sobre hojas y sombras tienen scores bajos, de 0,22 a 0,30. Esto respalda usar la confianza para decidir qué revisar.
- Hay cajas anidadas sobre un mismo tomate grande. Hace falta una fusión o una supresión mejor, y la verificación debe tratarlo.
- Los racimos de cherry pequeños y lejanos quedan sin detectar. Es el techo de recall ya medido en la fase 0.

**Label Studio.** `python -m autolabel.cli labelstudio --project <carpeta>` crea el proyecto y su plantilla, registra la carpeta de imágenes y carga las 120 tareas con las cajas como preanotaciones. Se verificó que las tareas existen, que 119 tienen predicción y que las imágenes se sirven. Una segunda carga sin `--replace` se rechaza para no duplicar.

**Detalles de uso**
- La imagen oficial 1.22.0 deshabilita los tokens heredados de la API. El cliente inicia sesión con usuario y contraseña (`LABEL_STUDIO_USERNAME` y `LABEL_STUDIO_PASSWORD`) y acepta un token si la organización lo permite. El `docker-compose.yml` no trae credenciales: la cuenta se crea la primera vez que se abre Label Studio, y esos datos se pasan por variables de entorno. Todos los puertos publicados, el de Label Studio, el del detector, el del servidor de calidad y el de Ollama, quedan solo en `127.0.0.1`, porque esos servicios no tienen autenticación.
- Para etiquetar por MCP hay que levantar el detector: `docker compose --profile services up -d mcp-detector`. Sin servidor, `--local` corre el detector en el mismo proceso desde la imagen GPU.

**Limitaciones**
- Las rutas de `images.txt` y `data.yaml` en la exportación YOLO son absolutas y de dentro del contenedor. Para entrenar en otra máquina hay que remapearlas.
- Aún no hay clasificación de madurez: en modo `agnostic` todas las cajas son una sola clase. Es la fase 3.
- No hay supresión de cajas anidadas entre sí más allá del NMS por IoU del detector.

### 8.4 Resultados de la fase 3: verificación y curador

**Qué hace.** Para cada caja del detector, el verificador recorta la imagen con un margen de contexto y le hace dos preguntas a Qwen3-VL 4B: si el recorte muestra ningún objeto, uno o un grupo, y a qué clase pertenece. Las respuestas son probabilidades leídas de los logprobs, no texto libre. Con ellas y el score del detector se calcula una **confianza combinada** con una regresión logística ajustada sobre el set dorado y guardada en `out/calibration.json`. Los prompts son plantillas con variables que se rellenan desde `project.yaml`, y la prueba de texto fijo sigue pasando.

Comandos:
- `python -m autolabel.cli curate --project <carpeta>` aplica el curador a las imágenes dudosas del filtro. Las que quedan en `keep` entran luego al etiquetado.
- `python -m autolabel.cli verify --project <carpeta>` verifica las cajas de `labels.json`, escribe `out/verification.json` y una vista previa coloreada (rojo: no es objeto, naranja: grupo, otros colores: clase). Es reanudable.

**Problema encontrado y resuelto.** La variante `qwen3-vl:4b` es de razonamiento: emite un bloque de pensamiento antes de responder aunque se le pida que no. Una primera versión del lector tomó las alternativas de ese primer token y devolvió probabilidades sin sentido. Se pasó a `qwen3-vl:4b-instruct` y el lector ahora falla con un mensaje claro si el modelo no responde con una opción. Una prueba automática cubre este caso.

**Clasificación de la madurez** (150 recortes de cajas verdaderas del set dorado, 50 por clase, azar sería 33%):

| Configuración | Exactitud | Recall verde | Recall semimaduro | Recall maduro | Tiempo por recorte |
|---|---|---|---|---|---|
| Solo texto | 0,62 | 0,86 | 0,42 | 0,58 | 1,0 s |
| Con 3 referencias por clase | 0,62 | 0,82 | 0,56 | 0,48 | 2,1 s |

- **Las referencias no mejoraron el total.** Movieron los errores: subió el semimaduro y bajó el maduro. Con 50 recortes por clase la incertidumbre ronda 7 puntos por clase, así que no se puede afirmar ni que ayuden ni que no. Las referencias de esta prueba salen de otro split de `laborotomato`, no del dominio de `tomato/`.
- **El semimaduro es la clase débil** y se confunde con sus dos vecinas, como se esperaba de una frontera difusa. El maduro se confunde bastante con verde.
- **La verdad del set dorado es ruidosa.** Algunas cajas de cualquier clase abarcan solo hojas o escenas amplias. En 46 de los 150 recortes el modelo no ve un solo objeto, y eso limita lo que se puede medir.

**Verificación de detecciones** (30 imágenes del set dorado, 411 cajas del detector: 309 aciertos y 102 falsos positivos):

| Señal | AUC | Precisión reteniendo 50% de los aciertos | Reteniendo 80% |
|---|---|---|---|
| Score del detector | 0,797 | 0,917 | 0,870 |
| Probabilidad de "un objeto" del VLM | 0,734 | 0,906 | 0,835 |
| **Confianza combinada** (validación cruzada) | **0,829** | **0,939** | **0,886** |

- **La mejora es real pero modesta**: unos 3 puntos de AUC y 1 a 2 de precisión. El modelo de 4B por sí solo separa peor que el detector, y combinados se complementan.
- **Cajas de grupo.** Se atacan con dos señales genéricas. La geometría marca como sospechosa una caja que contiene casi enteras a dos o más: 12 cajas (2,9%) con solo 8% de aciertos, frente a 77% en las demás. El VLM identifica un grupo en 11 de esas 12, y su probabilidad de grupo separa estas cajas del resto con AUC de 0,97. Probar un recorte más ajustado no cambió nada, así que el margen queda en 15%.
- **El VLM también marca como grupo cajas que sí son aciertos**, por ejemplo tomates apretados. Por eso su señal sola no basta para descartar y entra al modelo de confianza junto con la geometría.

**Curador** sobre las 105 imágenes dudosas de `tomato/`: 67 `keep`, 31 `drop` y 7 `unsure`, con un motivo en una frase para las apartadas. No hay verdad contra la cual medirlo, solo los motivos, que son coherentes (hojas desenfocadas, objetivo fuera de foco).

**Corrida real sobre `tomato/`** (120 imágenes, 597 cajas, unos 4,7 s por imagen, calibración del set dorado):

| Medida | Valor |
|---|---|
| Cajas que el VLM no ve como objeto (p>=0,5) | 175 (29%) |
| Cajas que ve como grupo (p>=0,5) | 55 (9%) |
| Clase según el VLM | maduro 328, verde 58, semimaduro 31, ninguna 180 |
| Confianza, percentiles 10 / 50 / 90 | 0,15 / 0,79 / 0,99 |

La vista previa se ve coherente: rojo sobre trozos de hoja, naranja sobre racimos y azul sobre tomates rojos claros.

**Limitaciones y lo que no se hizo**
- **La clasificación de madurez llega a 62%.** Sirve como señal para priorizar la revisión, no para aceptar sola. Un modelo mayor o una API probablemente mejoraría esto.
- **No se puede juzgar el reparto de clases en `tomato/`** (domina "maduro") porque no hay verdad para esas imágenes. Es una de las cosas que la revisión humana debe auditar.
- **La calibración sale de imágenes nítidas** del set dorado y puede no transferirse bien a las más borrosas de `tomato/`.
- **Las probabilidades del modelo son casi siempre 0 o 1.** La confianza combinada las suaviza, pero conviene no leerlas como probabilidades reales.
- **Diferido a la fase 4:** la revisión de objetos faltantes por imagen y la similitud por embeddings contra las referencias. No se implementaron en esta fase. La primera pide juzgar una imagen completa con cajas dibujadas, algo que un modelo de 4B hace mal, y conviene medirla dentro del bucle antes de confiar en ella.
- **Costo:** unos 1 s por caja. Verificar las 700 imágenes aceptadas de `tomato/` tomaría cerca de una hora.

### 8.5 Fase 4 (en curso): el bucle y el segundo proyecto

Esta sección registra lo construido y lo medido. El resultado principal es negativo en dos puntos y está en "Resultados del flujo completo".

**Correcciones de genericidad** (acordadas antes de empezar)
- **Sin cortes de nitidez por defecto.** Los valores 25 y 35 salían de `tomato/`. Ahora `blur_min` y `blur_ok` son opcionales y van juntos: sin ellos no se filtra por desenfoque, y el comando avisa.
- **`quality --calibrate`.** Sugiere los cortes y deja una hoja de muestras a lo largo de la distribución. Además de percentiles, calcula anclas absolutas: el puntaje que tendrían las imágenes más nítidas del propio conjunto con un desenfoque gaussiano de 1, 2 y 3 px. Así el corte se piensa como "aceptar hasta un desenfoque leve", y no como "descartar el 10% peor". En `tomatod`, donde todas las imágenes son nítidas, no apartó ninguna por desenfoque, que es lo correcto.
- **Regla de cajas de grupo por clase.** Una caja que contiene a dos o más ya solo cuenta las de su misma clase, así un objeto con partes de otra clase no se marca como grupo.

**Lo construido**
- **Mosaico en el detector** (`tiles`): detecta además por tiles solapados y funde con supresión por clase. Disponible como parámetro de estrategia y del servidor MCP.
- **Núcleo del bucle** (`autolabel/loop.py`): cada imagen lleva sus cajas con qué estrategia las encontró, lo que dijo la verificación y su estado. La fusión reconoce una caja ya encontrada, suma el acuerdo entre modelos distintos y no la verifica de nuevo. Cada caja se acepta, duda o rechaza con los umbrales de su clase. Una imagen sin dudas se acepta. Con dudas reintenta con la siguiente estrategia, salvo que el último intento no haya mejorado lo resuelto en `min_gain`. Al agotarse pasa a revisión con el motivo. Una imagen sin ninguna caja sigue intentando hasta agotar las estrategias, porque no hay mejora que medir.
- **Grafo** (`autolabel/graph.py`): filtro de calidad, curador, ciclo etiquetar y verificar, decisión y salidas. Los pasos largos procesan un lote por vez y vuelven a sí mismos, así cada lote queda en el checkpoint y una falla se retoma sin repetir lo hecho. Guarda una foto por ronda, el resultado final, el YOLO de lo aceptado y la lista de revisión.
- **Confianza de caja y de clase** (`Calibration`): la confianza final es la probabilidad de que la caja sea un objeto por la de que la clase sea correcta. Los umbrales de aceptación se sugieren con validación cruzada.
- **Comandos:** `run` ejecuta todo el flujo, y `labelstudio --from-run` carga solo las imágenes que quedaron en revisión.
- **`init`:** `python -m autolabel.cli init projects/mi-proyecto --classes a,b --scene "..."` crea la carpeta con un `project.yaml` de plantilla que pasa el esquema estricto. Los cortes de nitidez quedan sin definir a propósito, para calibrarlos con `quality --calibrate`. Rechaza pisar un proyecto existente salvo con `--force`.
- **Evaluación** (`eval/eval_pipeline.py`): mide cada ronda y el final contra la verdad, con una partición fija del set dorado (40% para calibrar y 60% para medir) para no evaluar sobre lo que sirvió de calibración.

**Segundo proyecto: `tomatod`.** Con clases de otros nombres (`unripe`, `semi-ripe`, `fully-ripe`), en inglés, y un `project.yaml` nuevo. Cargarlo y filtrarlo no requirió cambios de código. Lo único que se ajustó fue el armado del set dorado, que ahora acepta varios splits del dataset de origen. Set dorado de 100 imágenes a 1.000 px con 1.042 cajas: 42 imágenes para calibrar y 58 para medir.

**Mediciones del detector sobre `tomatod`** (58 imágenes de evaluación, 596 objetos, localizar con un solo prompt):

| Detector | AP50 | Mejor F1 | Precisión | Recall | Tiempo por imagen |
|---|---|---|---|---|---|
| tiny | 0,775 | 0,778 | 0,81 | 0,75 | 250 ms |
| tiny con mosaico 2 | 0,741 | 0,752 | 0,76 | 0,75 | 1.120 ms |
| base | 0,811 | 0,819 | 0,84 | 0,80 | 316 ms |

- **El mosaico no ayudó** en este dominio: rinde peor y es 4,5 veces más lento, porque los frutos de unos 50 px no son tan chicos como para necesitarlo. Queda implementado pero fuera de las estrategias de `tomatod`.
- **El modelo base sí mejora** al tiny, con casi el mismo costo. La lista de estrategias del proyecto quedó en tiny con umbral 0,2, luego base con 0,15 y luego base con 0,10.
- El detector rinde claramente mejor aquí que sobre el set dorado de `tomates` (AP50 0,55), lo que apunta a que la calidad de imagen pesa mucho.
- **Clasificar la madurez con un prompt por clase** sigue siendo flojo: AP50 de 0,32 para semi-ripe y 0,40 para fully-ripe con tiny.

**Sigue diferido:** la revisión de objetos faltantes por imagen y la similitud por embeddings. El bucle hoy no detecta objetos que ninguna estrategia encontró, así que una imagen puede aceptarse con objetos sin etiquetar.

**Calibración sobre `tomatod`** (42 imágenes, 826 cajas del detector, de las que 384 son aciertos):

| Señal | AUC | Precisión reteniendo 50% de los aciertos | Reteniendo 80% |
|---|---|---|---|
| Score del detector | 0,900 | 0,923 | 0,792 |
| Probabilidad de "un objeto" del VLM | 0,833 | 0,950 | 0,665 |
| Confianza de la caja, combinada | 0,907 | 0,932 | 0,824 |

- El modelo de visión **casi no mejora** la separación de aciertos y falsos positivos: de 0,900 a 0,907.
- La clase es correcta en 81% de las cajas bien localizadas, y eso acota todo lo demás.
- Con caja y clase correctas a la vez solo se alcanza **80% de precisión**: aceptar con confianza de 0,82 o más acepta el 12% de las cajas y el 28% de las correctas. Ningún umbral llega a 90%.

**Resultados del flujo completo** (58 imágenes de evaluación, 596 objetos, calibración de otra parte del set):

| | Ronda 1 | Final |
|---|---|---|
| Imágenes aceptadas solas | 2 de 54 | 2 de 54 |
| Imágenes a revisión | 52 | 52 |
| Objetos correctos que encontró alguna caja | 481 | 483 |
| Tiempo acumulado | 1.114 s | 2.024 s |

1. **El bucle no mejoró nada.** Las rondas 2 y 3 costaron el 45% del tiempo, 911 s, y aportaron 2 objetos correctos más. El modelo base y el tiny comparten los mismos puntos ciegos: ambos pierden los frutos muy chicos o tapados. El mecanismo funciona, pero estas estrategias no aportan diversidad. Casi todas las cajas las encontraron los dos modelos, así que el acuerdo entre modelos no distingue aciertos de errores.
2. **La aceptación por imagen es el cuello de botella.** Una imagen se acepta solo si ninguna de sus cajas es dudosa, y con unas 10 cajas por imagen casi siempre alguna lo es. Resultado: 2 imágenes aceptadas.
3. **Por caja sí funciona.** Las 239 cajas con confianza de 0,82 o más tienen 92% de localización correcta y **82% de acierto en caja y clase**, contra el 80% que daba la calibración. La calibración se transfiere bien a imágenes que no vio. Esas cajas resuelven el 33% de la verdad. Las 302 dudosas aciertan en localización el 63%, y las 1.459 rechazadas solo el 5%.
4. **El rechazo pierde objetos.** Con el umbral de revisión en 0,30 se descartaron 73 cajas verdaderas, el 15% de las cubiertas. La calibración sugería 0,17.
5. **El techo es el recall del detector.** El 19% de los objetos, 113 de 596, no lo encontró ninguna caja, y 24 más estaban en las imágenes apartadas por el filtro. Ninguna ronda cambió eso.

**Esfuerzo humano.** Contando cada acción igual, con 533 cajas a confirmar o borrar y 151 a dibujar, da 1,15 veces el trabajo de dibujar todo: no hay ahorro. Esa cuenta es injusta, porque confirmar o borrar una caja ya dibujada cuesta mucho menos que dibujarla. Si cuesta un cuarto, el trabajo baja a unas 0,48 veces, un ahorro cercano a la mitad. Ese cociente es una suposición y no una medición, y habría que medirlo con una persona.

**Carga a Label Studio con datos reales.** `labelstudio --from-run` creó el proyecto `tomatod-revision` con las 53 imágenes en revisión, cada una con sus cajas propuestas y los puntajes de confianza. El proyecto `tomates` de la fase 2 quedó intacto, con sus 120 tareas. Cada tarea trae las columnas `uncertain_boxes`, `confirmed_boxes` y `reason`, que sirven para ordenar y filtrar. Se verificó que las tareas existen, que las 53 tienen predicción y que las imágenes se sirven. **Sigue pendiente que una persona revise y se mida el tiempo.** Por defecto, `--from-run` usa el nombre del proyecto más `-revision`, para no mezclar con otras cargas, y `--title` lo cambia.

**Cambios aplicados a partir de esto**
- Decisiones **por caja**: cada imagen en revisión indica cuántas cajas están confirmadas, dudosas y rechazadas, la lista se ordena con las más dudosas primero y se genera una vista previa con colores. La evaluación mide por caja.
- **Una sola ronda** en `tomatod` y en la plantilla de `init`.
- Umbral de rechazo bajado de 0,30 a 0,17, el valor calibrado.

**Segunda medición, con esos cambios** (mismas 58 imágenes de evaluación, comparada con la corrida de 3 rondas y rechazo 0,30, guardada en `out/run_3rondas`)

| | 3 rondas, rechazo 0,30 | 1 ronda, rechazo 0,17 |
|---|---|---|
| Tiempo de trabajo | 2.024 s | 1.148 s |
| Objetos correctos encontrados por alguna caja | 483 | 481 |
| Cajas confirmadas | 239 | 243 |
| Acierto de lo confirmado, caja y clase | 82% | 82% |
| Cajas dudosas a revisar | 302 | 371 |
| Objetos que habría que dibujar | 186 | 170 |
| Esfuerzo si revisar cuesta 0,25 de dibujar | 0,439 | 0,441 |
| Esfuerzo si revisar cuesta lo mismo que dibujar | 0,819 | 0,908 |

- **La ronda única es mejor sin pérdida:** 43% menos de tiempo y los mismos objetos. Confirma que las rondas 2 y 3 no aportaban.
- **Bajar el rechazo no mejoró el esfuerzo con el costo supuesto.** Se muestran 69 cajas dudosas más, que cuestan en revisar lo que se ahorra en dibujar 16 objetos menos. Con un costo de revisar de 0,25 empata, y con costos más altos de revisar empeora. Solo ayuda si revisar cuesta mucho menos que dibujar. Ese cociente es la suposición clave y todavía no está medido con una persona.
- **Un defecto encontrado en la medición:** `--restart` no borraba las fotos de ronda de la corrida anterior, y la evaluación mostraba columnas viejas como si fueran de la corrida nueva. Está corregido, con prueba.
- **Lo que queda:** el recall del detector, con 113 objetos que ninguna caja encontró, y la clase, que acierta 81% entre las cajas bien localizadas.

## 9. Métricas

- Tasa de autoaceptación.
- Precisión de lo autoaceptado (objetivo 95% o más sobre el set dorado).
- Recall de frutos.
- Minutos humanos por 100 imágenes frente a etiquetar desde cero.
- Imágenes buenas descartadas por error por el curador.

## 10. Riesgos

- **Objetos chicos y tapados:** Grounding DINO puede fallar con tomates de pocos píxeles en 400x500. Mitigación: mosaico (tiling) y escalado de la imagen antes de detectar.
- **Tool-calling débil en modelos chicos:** ya mitigado con salida estructurada y sin bucle libre de herramientas.
- **Bucle sin fin o sesgado:** el ciclo etiquetado ↔ verificación está acotado por intentos, mejora mínima y tiempo. El verificador usa un modelo distinto del etiquetador para no validar sus propios errores, y la precisión de lo autoaceptado se mide contra el set dorado.
- **Modelos de razonamiento como VLM:** devuelven un bloque de pensamiento y no permiten leer probabilidades. Hay que usar variantes de instrucción. El adaptador falla con un mensaje claro si ocurre.
- **Clasificación débil de madurez con un modelo de 4B:** 62% de exactitud. Es una señal de prioridad, no una decisión. Un modelo mayor o una API es la mejora más probable.
- **Costo del bucle en una GPU de 8 GB:** cada estrategia puede exigir otro modelo. Se agrupan las imágenes por estrategia en cada vuelta para cargar cada modelo una sola vez.
- **VRAM justa:** mitigado con ejecución por etapas y un solo modelo cargado a la vez.
- **Confianza mal calibrada:** se calibra con el set dorado y se audita una muestra de lo autoaceptado.
- **Licencias:** SAM 3 es propia de Meta. Revisar antes de uso comercial.
- **Dominio distinto entre el set dorado y `tomato/`:** `laborotomato` son fotos nítidas de celular y `tomato/` son más borrosas y con frutos más chicos. Los números del set dorado pueden ser optimistas. Se cubre con la reducción de tamaño y con la auditoría humana en `tomato/`.
- **Distinguir maduración por color:** verde, semimaduro y maduro son clases con fronteras difusas. Se espera confusión entre vecinas, sobre todo semimaduro. La comparación directa contra dos pasos de la sección 5 apunta a esto.
- **Clases desbalanceadas:** domina la clase verde. Las métricas se reportan por clase, no solo en promedio.
- **Orientación EXIF:** una imagen sin girar produce cajas desalineadas sin dar error. Mitigado con el cargador común y la comprobación de proporciones. Resuelto en la fase 2: el servidor del detector y las exportaciones a YOLO y a Label Studio usan las coordenadas de la imagen ya girada.
- **Recall del detector:** la fase 0 mostró que deja cerca de la mitad de los frutos sin detectar. Hay que atacarlo en la fase 7 y ser claro en que lo no detectado no llega al humano salvo que se revise la imagen completa.
- **Arrastre del proyecto de prueba:** al desarrollar con tomates es fácil dejar supuestos fijos (tamaños, colores, umbrales). Se cubre con la prueba de texto fijo y el segundo proyecto de la sección 6.3.

## 11. Pendiente de confirmar

1. Si el uso será comercial. Afecta la licencia de SAM 3 y los datasets, que son no comerciales.
2. Formato de salida preferido (YOLO, COCO o ambos).
3. Qué hacer con el recall del detector. El bucle ayuda: la verificación revisa si faltan objetos y reintenta con otras estrategias. Falta decidir qué pasa con lo que aun así queda sin detectar: aceptarlo o mandar la imagen completa a revisión.

## Fuentes

- [Roboflow Vision Evals, detección](https://benchmarklist.com/benchmarks/roboflow_vision_evals_detection/)
- [Grounding DINO vs SAM 3](https://playground.roboflow.com/models/compare/grounding-dino-vs-sam-3)
- [Open-vocabulary detection: Grounding DINO, Florence-2, RT-DETR, RF-DETR](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr)
- [LangGraph vs Strands (2026)](https://www.cipherprojects.com/blog/posts/langgraph-vs-strands-agents-2026/)
