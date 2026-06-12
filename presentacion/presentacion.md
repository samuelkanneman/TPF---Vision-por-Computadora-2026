---
marp: true
theme: default
paginate: true
size: 16:9
footer: "Trabajo Final — Visión por Computadora 2026"
---

<!--
BORRADOR DE PRESENTACIÓN (14 slides, según estructura_presentacion.md)

Cómo usarlo:
  - Opción A (Marp): instalar la extensión "Marp for VS Code" y exportar a PDF/PPTX,
    o con CLI: npx @marp-team/marp-cli presentacion.md --pptx
  - Opción B: copiar el contenido slide por slide a Google Slides / PowerPoint.

Los bloques [INSERTAR: ...] se completan con el material que sale del
entrenamiento y la inferencia (ver AGENTS.md sección 5.3):
  - runs/detect/train/results.png, confusion_matrix.png, PR_curve.png
  - runs/detect/predict_test/  (predicciones sobre el set de validación)
  - frames de videos/output/resultado_video*.mp4
-->

# Detección, Segmentación y Pose con YOLO26

## Trabajo Final — Visión por Computadora 2026

**Samuel Kanneman · Federico Ford**

Junio 2026

---

# Introducción

- **YOLO** (*You Only Look Once*): familia de modelos de detección en tiempo real — una sola pasada de red para localizar y clasificar objetos.
- **YOLO26** (Ultralytics, enero 2026): inferencia *end-to-end* sin NMS, más liviano y rápido en CPU/edge, con cabezales para detección, segmentación y pose.
- **Objetivo del trabajo**: integrar las tres tareas en un único pipeline de video.
- **Desafío**: detectar objetos que **no existen en COCO** → hay que entrenar un modelo propio con transfer learning.

---

# Objetivos

1. **Entrenar** YOLO26 con 3 clases personalizadas (no incluidas en las 80 de COCO).
2. **Integrar** en un solo pipeline:
   - Detección custom (modelo propio)
   - Segmentación de instancias COCO (preentrenado)
   - Estimación de pose humana (preentrenado)
3. **Demostrar** el sistema procesando videos reales, con métricas y FPS.

---

# Dataset

| | |
|---|---|
| **Clases (Set A)** | `0: factura` · `1: mate` · `2: termo` |
| **¿Por qué?** | Objetos cotidianos argentinos, ausentes en COCO |
| **Imágenes** | 155 (124 train / 31 valid, split 80/20) |
| **Instancias (train)** | 172 factura · 55 mate · 35 termo |
| **Etiquetado** | Roboflow (polígonos YOLO), licencia CC BY 4.0 |

[INSERTAR: captura de Roboflow con el dataset etiquetado]
[INSERTAR: salida del sanity check visual del train notebook (celda 3.1)]

---

# Configuración del entrenamiento

- **Modelo base**: `yolo26n.pt` (nano) — *transfer learning* desde COCO
- **Épocas**: 80 (con *early stopping*, `patience=15`)
- **Batch size**: 16 · **Image size**: 640
- **Optimizador**: AdamW, `lr0 = 0.001`
- **Augmentations**: las default de Ultralytics (mosaic, flips, HSV, etc.)
- **Hardware**: Google Colab (GPU T4)

---

# Métricas de entrenamiento

| Métrica | Objetivo | Obtenido |
|---------|----------|----------|
| mAP50 | ≥ 0.70 | [INSERTAR] |
| mAP50-95 | ≥ 0.55 | [INSERTAR] |
| Precision | ≥ 0.80 | [INSERTAR] |
| Recall | ≥ 0.70 | [INSERTAR] |

[INSERTAR: runs/detect/train/results.png — curvas de loss y métricas]
[INSERTAR: runs/detect/train/confusion_matrix.png]
[INSERTAR: runs/detect/train/PR_curve.png]

---

# Resultados del modelo custom

**Predicciones sobre el set de validación** (`runs/detect/predict_test/`):

[INSERTAR: 2-3 ejemplos de predicciones correctas]

[INSERTAR: 1-2 ejemplos de errores o falsos positivos]

**Discusión**: qué clase funcionó mejor / peor y por qué
(pista: el desbalance 172/55/35 sugiere que `termo` puede ser la más difícil).

---

# Arquitectura del pipeline de inferencia

```
                    ┌──────────────────────────┐
                    │   Frame del video        │
                    └────────────┬─────────────┘
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
 ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
 │ yolo26n-seg.pt  │   │ yolo26n-pose.pt  │   │    best.pt      │
 │ Segmentación    │   │ Pose (COCO-17)   │   │ Detección custom│
 │ COCO            │   │                  │   │ factura/mate/   │
 │ (sin person ni  │   │                  │   │ termo           │
 │  clases custom) │   │                  │   │                 │
 └────────┬────────┘   └────────┬─────────┘   └────────┬────────┘
          └──────────────────────┼──────────────────────┘
                                 ▼
                  ┌───────────────────────────┐
                  │  utils.annotate_frame()   │
                  │  máscaras + skeletons +   │
                  │  bboxes + overlay (FPS,   │
                  │  conteos)                 │
                  └────────────┬──────────────┘
                               ▼
                  videos/output/resultado_*.mp4
```

---

# Lógica de exclusión

**¿Por qué excluimos clases de la segmentación COCO?**

- `person` → ya la representa el modelo de **pose** (skeleton); dibujar también la máscara duplicaría la anotación y ensuciaría el frame.
- Clases custom → las dibuja el detector **custom** con bbox + confianza; si COCO segmentara algo parecido (ej. `cup`, `bottle`), habría doble anotación contradictoria.

**Resultado**: cada objeto es responsabilidad de exactamente **un** modelo →
frame limpio y sin solapamientos.

---

# Demostración: video procesado

[INSERTAR: video resultado_video1.mp4 embebido, o GIF corto si el archivo es pesado]

Qué observar:
- 🟦 Máscaras de segmentación sobre objetos COCO
- 🦴 Skeletons sobre las personas
- 🟥 Bounding boxes con confianza sobre factura / mate / termo
- 📊 Overlay con FPS y conteos por categoría

---

# Ejemplos de frames

[INSERTAR: 3-4 frames destacados de videos/output/ — los genera la última celda del inference notebook]

[INSERTAR: comparativa antes / después del mismo frame]

---

# Dificultades encontradas

- **Nombres y orden de clases**: el export de Roboflow definía `0: bolleria, 1: mate, 2: termo`, distinto a la config inicial del repo → se corrigió `data.yaml` respetando los índices (si no, las clases quedaban cruzadas).
- **Formato de labels**: Roboflow exportó polígonos (segmentación) en vez de bboxes → Ultralytics los convierte automáticamente para detección.
- **Dataset desbalanceado**: 172 facturas vs 35 termos en train.
- **Imágenes 512×512 con stretch** (preprocesado de Roboflow) → pérdida de relación de aspecto.
- **Trabajo distribuido**: entrenamiento en Colab + repo en GitHub + videos por Drive.
- [INSERTAR: dificultades de performance/FPS observadas en la inferencia]

---

# Conclusiones

- ✅ Pipeline funcional que integra **3 tareas de visión** en un solo paso de video.
- ✅ **Transfer learning** permite entrenar un detector útil con solo ~150 imágenes.
- ✅ La **calidad y consistencia del dataset** (orden de clases, formato de labels) importa tanto como el modelo.

**Líneas futuras**: más datos y balanceo de clases · tracking entre frames (ByteTrack) · export a ONNX/TensorRT para más FPS · más clases custom.

---

<!-- _class: lead -->

# ¡Muchas gracias!

## ¿Preguntas?

Repo: `github.com/samuelkanneman/TPF---Vision-por-Computadora-2026`
