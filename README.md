# 🎓 Trabajo Final - Visión por Computadora 2026

**Detección, Segmentación y Estimación de Pose utilizando YOLO26**

Proyecto integrador que aplica modelos de la familia YOLO para detectar clases personalizadas, segmentar objetos COCO y estimar la pose humana, todo en un único pipeline de procesamiento de video.

---

## 📁 Estructura del proyecto

```
Trabajo Final/
├── data/
│   ├── data.yaml                  # Configuración del dataset (apunta a mate-termo/)
│   └── mate-termo/                # Dataset custom (export de Roboflow, 155 imágenes)
│       ├── train/
│       │   ├── images/            # 124 imágenes
│       │   └── labels/            # Labels YOLO (polígonos)
│       └── valid/
│           ├── images/            # 31 imágenes
│           └── labels/
├── training/
│   └── train_notebook.ipynb       # Notebook de entrenamiento
├── inference/
│   ├── inference_notebook.ipynb   # Notebook de inferencia sobre video
│   └── utils.py                   # Funciones auxiliares de dibujo
├── runs/detect/train/weights/
│   ├── best.pt                    # Pesos entrenados (entregable)
│   └── last.pt
├── videos/
│   ├── input/                     # Videos de entrada (2-3 de ~20s)
│   └── output/                    # Video procesado final
├── presentacion/
│   └── estructura_presentacion.md # Outline de la presentación
├── Proyecto Integrador Visión por Computadora - 2026.pdf
├── requirements.txt
└── README.md
```

---

## 🎯 Clases personalizadas (Set A)

| ID | Clase   | Descripción                            |
|----|---------|----------------------------------------|
| 0  | factura | Factura / pan dulce de panadería       |
| 1  | mate    | Mate (recipiente para yerba mate)      |
| 2  | termo   | Termo para agua caliente               |

> ⚠️ Este es el **orden del export de Roboflow** (donde la clase 0 se llama `bolleria`; se renombró a `factura` manteniendo los índices). No reordenar `names` en `data/data.yaml`: los labels usan estos índices.

### 📊 Dataset

- **155 imágenes** (124 train / 31 valid, split 80/20), etiquetadas con [Roboflow](https://universe.roboflow.com/federicos-workspace-5cgiu/mate-termo-factura/dataset/1) (licencia CC BY 4.0).
- Instancias en train: 208 factura · 106 mate · 71 termo (en valid: 22 · 29 · 19).
- Labels en formato polígono YOLO (export de segmentación); ultralytics los convierte automáticamente a bounding boxes para entrenar detección.
- El dataset está **incluido en el repo** (`data/mate-termo/`, ~7 MB): con clonar alcanza, también en Colab.

---

## 🚀 Quick start

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Dataset

Ya está incluido en `data/mate-termo/` (export de Roboflow). El notebook de entrenamiento lo valida, muestra un **sanity check visual** del etiquetado y genera la config con rutas absolutas automáticamente.

### 3. Entrenar el modelo

Abrir y ejecutar `training/train_notebook.ipynb` (Run All). Funciona en **local o Google Colab** (la primera celda detecta el entorno). Al finalizar genera `runs/detect/train/weights/best.pt` y lo copia a la raíz del repo.

### 4. Procesar los videos

1. Tener los videos en `videos/input/` (los 3 videos ya están commiteados en el repo).
2. Abrir y ejecutar `inference/inference_notebook.ipynb` (Run All): procesa **todos** los videos de `videos/input/` con los 3 modelos en paralelo.
3. Los videos procesados quedan en `videos/output/resultado_<nombre>.mp4`.

---

## 🧪 Modelo y configuración usada

- **Modelo base**: `yolo26n.pt` (transfer learning) — [YOLO26](https://docs.ultralytics.com/models/yolo26) está disponible en `ultralytics >= 8.4` (enero 2026); el notebook tiene fallback automático a `yolo11n.pt`.
- **Épocas**: 150 con early stopping (`patience=30`) — el run final cortó en la época 119 (mejor: la 89)
- **Batch size**: 16
- **Image size**: 640
- **Optimizador**: AdamW
- **Augmentations**: Activadas (default Ultralytics)

## 📈 Resultados finales

| Métrica | Objetivo | Obtenido |
|---------|----------|----------|
| mAP50 (global) | ≥ 0.70 | **0.57** |
| mAP50 — termo | | 0.76 |
| mAP50 — mate | | 0.66 |
| mAP50 — factura | | 0.29 |
| Precision / Recall | 0.80 / 0.70 | 0.56 / 0.62 |
| FPS del pipeline (T4) | ≥ 5 (ideal 15) | **12.3 – 17.5** |

Historial de experimentos (mismo dataset):

| Run | Modelo | Épocas | mAP50 | Conclusión |
|-----|--------|--------|-------|------------|
| v1 | yolo26n | 80 | 0.59 | la curva seguía subiendo |
| v2 | yolo26s | 150 (stop 78) | 0.53 | **overfit** — el small no ayuda con 124 imágenes |
| v3 (final) | yolo26n | 150 (stop 119) | 0.57 | mejor recall (0.62); nivel estable del nano |

> Análisis: casi no hay confusión entre clases — los errores son objetos no detectados (recall), concentrados en `factura` (objetos chicos, agrupados en bandejas). Detalle en `presentacion/presentacion.md`.

> Ajustar estos valores según VRAM disponible y tiempo de entrenamiento.

---

## 📦 Entregables

- [x] Notebooks (training + inference)
- [x] Scripts (`utils.py`)
- [x] Dataset etiquetado (`data/mate-termo/`, 155 imágenes)
- [x] `data/data.yaml`
- [x] Videos originales (`videos/input/`, 3 videos)
- [x] `best.pt` (pesos entrenados, yolo26n — en la raíz del repo)
- [x] Videos procesados finales (`videos/output/resultado_video{1,2,3}.mp4`)
- [x] Presentación (`presentacion/presentacion.md`, 15 slides con métricas y material en `presentacion/img/` — exportar a PPTX/PDF con Marp)

---

## 📅 Fechas importantes

- **Entrega notebooks**: Lunes 15 de junio
- **Presentación**: Primer día hábil de la última semana de cursada

---

## ⚠️ Notas

- **YOLO26** fue lanzado por Ultralytics en enero 2026 e incluye los checkpoints usados en este trabajo: `yolo26n.pt` (detección), `yolo26n-seg.pt` (segmentación) y `yolo26n-pose.pt` (pose). Requiere `ultralytics >= 8.4` (los notebooks hacen `pip install -U`).
- **Clases excluidas en segmentación COCO**: `person` y los nombres de las clases custom (mate, termo, factura). Las personas se detectan con el modelo de **Pose** y las clases custom con el modelo **Detection** entrenado.
