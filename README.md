# autolabel: etiquetado automático de imágenes para detección de objetos

Dado un conjunto de imágenes y las clases que querés, el flujo aparta las imágenes inservibles, dibuja las cajas con un detector, las verifica con un modelo de visión y deja para revisión humana solo lo dudoso. Todo corre en tu máquina, sin API keys.

> **Nota:** este repositorio no incluye imágenes. Los datasets de tomates con los que se desarrolló y probó el flujo pesan más de 2 GB, y sus licencias son de uso no comercial, así que no se subieron. Las carpetas de `projects/` solo traen su `project.yaml` como ejemplo de configuración y no corren sin imágenes. Para probar el flujo, creá tu propio proyecto con tus imágenes siguiendo los pasos de abajo.

## Qué necesitás

- Una notebook o PC con **GPU NVIDIA de 8 GB o más** y **Docker Desktop** (en Windows, con WSL2).
- Unos **40 GB libres** y conexión a internet la primera vez, para bajar imágenes y modelos.
- Un **set de referencia**: unas 100 de tus imágenes con las cajas ya dibujadas. Se usa para calibrar la confianza del sistema. Formato más abajo.

Comprobá que Docker ve la GPU:

```
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
```

## 1. Instalación (una sola vez por máquina)

Desde la carpeta del proyecto:

```
docker compose build gpu light
docker compose --profile vlm up -d ollama
docker compose exec ollama ollama pull qwen3-vl:4b-instruct
```

La construcción tarda unos 10 minutos. Los modelos del detector se descargan solos la primera vez que se usan.

Si Docker se queja de que falta la carpeta `dataset-tomates`, creala vacía: `mkdir dataset-tomates`.

## 2. Crear tu proyecto

Cada proyecto es una carpeta dentro de `projects/` con un `project.yaml`.

```
docker compose run --rm light python -m autolabel.cli init projects/mi-proyecto --classes casco,chaleco --scene "fotos de una obra de construcción"
```

Después:

1. **Copiá tus imágenes** a `projects/mi-proyecto/images/`.
2. **Editá `projects/mi-proyecto/project.yaml`** y completá, para cada clase:
   - `prompt`: cómo se llama el objeto, en inglés y descriptivo (por ejemplo `"yellow safety helmet"`).
   - `definition`: cómo reconocerlo, en una frase.
   - `not_this`: cosas parecidas que no son esa clase.
3. En el bloque `detector`, completá **`agnostic_prompt`** con una palabra que agrupe todas las clases (por ejemplo `"helmet"`).

## 3. Fijar el corte de imágenes borrosas

```
docker compose run --rm light python -m autolabel.cli quality --project projects/mi-proyecto --calibrate
```

Muestra los cortes sugeridos y guarda una hoja de muestras en `projects/mi-proyecto/out/quality_quantiles.jpg`, de la menos a la más nítida. Mirala, confirmá que los cortes tienen sentido y copiá `blur_min` y `blur_ok` al bloque `quality` del YAML (hay un ejemplo comentado). Si no los definís, no se filtra por desenfoque.

## 4. Calibrar con tu set de referencia

Guardá las etiquetas de referencia en `projects/mi-proyecto/gold.json`. Las imágenes tienen que estar en `images/`:

```json
{
  "foto1.jpg": {
    "size": [1000, 750],
    "boxes": [
      {"cls": "casco",   "box": [120, 40, 210, 130]},
      {"cls": "chaleco", "box": [300, 200, 420, 380]}
    ]
  }
}
```

Las cajas son `[x1, y1, x2, y2]` en píxeles de la imagen, ya girada si la cámara la guardó rotada, y `cls` debe coincidir con el nombre de una clase del YAML.

Primero el detector, y después el verificador, que necesita Ollama:

```
docker compose run --rm gpu python -m eval.eval_detector --project projects/mi-proyecto --part calib
docker compose --profile vlm up -d ollama
docker compose run --rm light python -m eval.eval_verifier --project projects/mi-proyecto --part calib
```

Esto genera `out/calibration.json` y al final imprime **umbrales sugeridos**. Copiá `auto_accept` y `review` a `thresholds` de cada clase en el YAML. Si el sistema indica que ningún umbral alcanza buena precisión, significa que casi todo irá a revisión humana.

## 5. Correr el flujo

```
docker compose --profile services --profile vlm up -d mcp-detector ollama
docker compose run --rm light python -m autolabel.cli run --project projects/mi-proyecto
```

El flujo filtra las imágenes, etiqueta, verifica y reintenta con modelos más potentes lo que quede dudoso. Puede tardar horas con muchas imágenes. Si se corta, volvé a correr el mismo comando y retoma donde quedó. Para empezar de cero, agregá `--restart`.

Para probar con pocas imágenes primero, agregá `--limit 20`.

**Resultados** en `projects/mi-proyecto/out/run/`:

| Archivo | Qué contiene |
|---|---|
| `yolo/` | Etiquetas en formato YOLO de las imágenes que se aceptaron solas |
| `review.json` | Imágenes que necesitan revisión humana, con el motivo |
| `final.json` | Todo el detalle por imagen y por caja |

## 6. Revisar lo dudoso

```
docker compose --profile review up -d label-studio
```

La primera vez, abrí `http://127.0.0.1:8080` y creá tu cuenta. Elegí tu propio correo y una contraseña. El sistema no trae ninguna cuenta ni contraseña por defecto, y el puerto solo es accesible desde tu máquina. Después cargá las imágenes a revisar con las credenciales que acabás de crear:

```
docker compose run --rm -e LABEL_STUDIO_USERNAME=tu-correo -e LABEL_STUDIO_PASSWORD=tu-contraseña light python -m autolabel.cli labelstudio --project projects/mi-proyecto --from-run
```

En Label Studio abrí el proyecto `mi-proyecto-revision`. Es un proyecto aparte, para no mezclar estas tareas con otras. Solo cargan las imágenes que quedaron en revisión, con las cajas propuestas ya dibujadas.

Cada tarea trae las columnas `uncertain_boxes` y `confirmed_boxes`. Ordená por `uncertain_boxes` para empezar por las que tienen más cajas dudosas. Las cajas dudosas son las de menor puntaje. Podés cambiar el nombre del proyecto con `--title`. Si repetís el comando sobre un proyecto que ya tiene tareas, pide `--replace`.

## Problemas frecuentes

| Síntoma | Qué hacer |
|---|---|
| Docker no ve la GPU | Actualizá el driver de NVIDIA y reiniciá Docker Desktop |
| `model not found` en Ollama | Repetí el `ollama pull` del paso 1 |
| No conecta con `mcp-detector` | Levantalo con el comando del paso 5 |
| Se queda sin memoria de GPU | No corras `eval_detector` mientras `mcp-detector` y `ollama` están activos: `docker compose stop mcp-detector` |
| Etiqueta clases equivocadas | Mejorá `prompt` y `definition` de cada clase en el YAML y repetí el paso 4 |

## Tests

```
docker compose run --rm light pytest
```
