# AGENTS.md

> Instrucciones operativas para agentes de IA y colaboradores humanos que trabajen
> en este repositorio. Leer antes de modificar código, abrir PRs o ejecutar
> entrenamiento/inferencia.

---

## 1. Resumen del proyecto

**Trabajo Final — Visión por Computadora 2026 (UGR)**

Proyecto integrador que aplica modelos de la familia **YOLO** para resolver
tres tareas en un único pipeline de procesamiento de video:

1. **Detección de clases personalizadas** (modelo fine-tuneado del grupo)
2. **Segmentación de instancias COCO** (modelo preentrenado, excluye `person` y las clases custom)
3. **Estimación de pose humana** (modelo preentrenado)

**Clases custom (Set A — gastronomía argentina):**

| ID | Nombre      | Descripción                            |
|----|-------------|----------------------------------------|
| 0  | `factura`   | Factura / pan dulce de panadería       |
| 1  | `mate`      | Mate (recipiente para yerba mate)      |
| 2  | `termo`     | Termo para agua caliente               |

> **Importante:** este es el orden del export de Roboflow (`data/mate-termo/`),
> donde la clase 0 se llama `bolleria`. En `data/data.yaml` se renombró a
> `factura` **manteniendo el orden 0/1/2** — los índices de los labels no se
> tocan, solo el nombre. NO reordenar `names`.

---

## 2. Tech stack

- **Python** ≥ 3.10
- **Ultralytics** (incluye YOLO11 / YOLO26 según disponibilidad)
- **PyTorch** + **torchvision** (backend de YOLO)
- **OpenCV** (`cv2`) para I/O de video y dibujo
- **Pillow**, **numpy**, **PyYAML**, **tqdm**
- **Jupyter** para los notebooks (no scripts `.py` salvo `utils.py`)

Dependencias completas en [`requirements.txt`](./requirements.txt).

---

## 3. Estructura del repositorio

```
.
├── AGENTS.md                          # este archivo
├── README.md                          # documentación pública
├── TAREAS.md                          # roadmap operativo (entrega 15/06)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── data.yaml                      # config del dataset (apunta a mate-termo/)
│   ├── data_runtime.yaml              # (gitignored) generado por el train notebook
│   ├── mate-termo/                    # export de Roboflow (se commitea, ~7 MB)
│   │   ├── train/{images,labels}/     # 124 imágenes + labels (polígonos YOLO)
│   │   └── valid/{images,labels}/     # 31 imágenes + labels
│   └── images|labels/{train,val}/     # vacíos (estructura legacy, solo .gitkeep)
│
├── training/
│   └── train_notebook.ipynb           # celdas de entrenamiento
│
├── inference/
│   ├── inference_notebook.ipynb        # pipeline integrado de 3 modelos
│   └── utils.py                       # funciones de dibujo (bboxes, masks, pose)
│
├── videos/
│   ├── input/                         # videos a procesar (3 videos, ~20s c/u)
│   └── output/                        # videos anotados resultantes
│
├── presentacion/
│   └── estructura_presentacion.md      # outline de las 14 slides
│
└── runs/                              # (gitignored) outputs de Ultralytics
    └── detect/train/
        ├── best.pt                    # entregable principal del training
        ├── last.pt
        ├── results.png                # gráficas de loss/métricas
        ├── confusion_matrix.png
        └── PR_curve.png
```

---

## 4. Setup

```bash
# Clonar e instalar
git clone <repo>
cd TPF---Vision-por-Computadora-2026
pip install -r requirements.txt
```

> **YOLO26 existe desde enero 2026** (incluye `yolo26n.pt`, `yolo26n-seg.pt` y
> `yolo26n-pose.pt`) — requiere `ultralytics >= 8.4`. Los notebooks hacen
> `pip install -U ultralytics` y tienen fallback automático a `yolo11n.pt`
> por si la descarga falla. Docs: https://docs.ultralytics.com/models/yolo26

---

## 5. Workflows principales

### 5.1. Entrenar el modelo custom

1. Abrir [`training/train_notebook.ipynb`](./training/train_notebook.ipynb) y correr
   la celda 0 (detecta Colab o local y se para en la raíz del repo).
2. Verificar GPU disponible (sección 2).
3. Verificar dataset (sección 3): debe contar 124 train imgs + 31 valid imgs
   en `data/mate-termo/`. Correr el **sanity check visual** (3.1) y la
   generación de `data/data_runtime.yaml` (3.2) — el entrenamiento usa ese
   yaml con rutas absolutas.
4. Hiperparámetros (sección 5):
   - `MODEL_NAME = 'yolo26n.pt'` (fallback automático a `yolo11n.pt`)
   - `data=data/data_runtime.yaml` (autogenerado)
   - `epochs=80`, `batch=16` (bajar a 8 si OOM), `imgsz=640`
   - `optimizer='AdamW'`, `patience=15`
5. Correr. Resultado en `runs/detect/train/weights/best.pt`.
6. Copiar `best.pt` a la raíz (sección 8 del notebook ya lo hace).

**Métricas objetivo** (de TAREAS.md):

| Métrica                    | Mínimo  | Ideal   |
|----------------------------|---------|---------|
| mAP50 (clases custom)      | ≥ 0.50  | ≥ 0.75  |
| mAP50-95 (clases custom)    | ≥ 0.35  | ≥ 0.55  |
| Precision (clases custom)   | ≥ 0.60  | ≥ 0.80  |
| Recall (clases custom)      | ≥ 0.50  | ≥ 0.70  |
| FPS del pipeline            | ≥ 5     | ≥ 15    |

### 5.2. Correr inferencia sobre un video

1. Copiar videos a `videos/input/`. Ya hay 3 (video1/2/3.mp4) — están
   gitignoreados, en Colab llegan por el Drive compartido.
2. Tener `best.pt` en la raíz (lo genera el notebook de training).
3. Abrir [`inference/inference_notebook.ipynb`](./inference/inference_notebook.ipynb)
   y correr todas las celdas: procesa **todos** los videos de `videos/input/`
   automáticamente y guarda `resultado_<nombre>.mp4` en `videos/output/`.
4. Para debug rápido, poner `MAX_FRAMES = 100` en la celda de procesamiento.

El pipeline aplica los 3 modelos en paralelo a cada frame y dibuja:
- Bounding boxes de las 3 clases custom (con label + confianza)
- Máscaras de segmentación COCO (excluyendo `person` y clases custom)
- Skeletons de pose humana (COCO-17 keypoints)

### 5.3. Capturar material para la presentación

- `runs/detect/train/results.png` → gráficas de loss/métricas (slide 6)
- `runs/detect/train/confusion_matrix.png` → matriz de confusión (slide 6)
- `runs/detect/train/PR_curve.png` → curvas PR (slide 6)
- Frames de `videos/output/video*.mp4` → capturas para slides 7, 10, 11

---

## 6. Convenciones del repo

### 6.1. Git

- **Branch principal:** `main` (protegido, requiere PR)
- **Mensajes de commit** en español, prefijos convencionales:
  - `feat:` nueva funcionalidad
  - `fix:` corrección de bug
  - `chore:` tareas de mantenimiento (limpieza, .gitignore, etc.)
  - `docs:` cambios solo en documentación
  - `refactor:` cambios que no agregan features ni fixean bugs
- **NO commitear** (ver `.gitignore`):
  - `runs/` (outputs de Ultralytics)
  - `*.pt` (modelos entrenados)
  - `videos/output/*` (videos procesados, son pesados)
  - `__pycache__/`, `.venv/`, etc.

### 6.2. Notebooks

- **Limpiar outputs antes de commitear**: `Kernel → Restart & Clear All Outputs`
- Mantener las celdas en orden lógico: imports → setup → config → ejecución → resultados
- Una sola cosa por celda, con markdown explicativo entre bloques

### 6.3. Python (`utils.py`)

- Funciones puras cuando sea posible
- Type hints en firmas
- Docstrings con `Args:` / `Returns:` (estilo Google)
- Constantes en MAYÚSCULAS al tope del archivo (ej. `SKELETON_PAIRS`)

### 6.4. Datos / labels

- Los labels de `data/mate-termo/` están en **formato polígono** de YOLO
  (`class_id x1 y1 x2 y2 ...`, normalizado 0-1) porque Roboflow los exportó
  como segmentación. Ultralytics los convierte solo a bboxes para `task=detect`.
- Orden de clases: `0=factura` (en Roboflow se llama `bolleria`), `1=mate`,
  `2=termo`. **No reordenar** `names` en `data/data.yaml`.
- Una imagen sin objetos → archivo `.txt` **vacío** (no borrarlo)
- No usar caracteres especiales ni espacios en nombres de archivo
- Si se reentrena, NO borrar `runs/detect/train/` hasta confirmar que el nuevo
  entrenamiento dio mejores métricas (backup por las dudas)

---

## 7. Troubleshooting común

| Problema                                         | Solución                                              |
|--------------------------------------------------|-------------------------------------------------------|
| `ModuleNotFoundError: No module named 'ultralytics'` | `pip install ultralytics`                          |
| `FileNotFoundError: best.pt`                    | Correr primero el notebook de training               |
| `VIDEO_PATH not found`                          | Verificar que el video está en `videos/input/`        |
| FPS muy bajo (<5)                               | Usar `yolo26n` (no `m`), reducir `imgsz` a 320, usar GPU |
| No detecta clases custom                        | Verificar que `best.pt` está bien entrenado (mAP > 0.5) |
| No detecta personas                             | Usar `yolo26n-pose.pt`, no `yolo26n.pt`              |
| CUDA Out of Memory                               | Bajar `batch` de 16 a 8 (o 4)                        |
| `Dataset not found` al entrenar                  | Correr la celda 3.2 (genera `data_runtime.yaml` con rutas absolutas) |
| Clases cruzadas (mate detectado como termo)      | Revisar el orden de `names` en `data/data.yaml` (correcto: 0=factura, 1=mate, 2=termo) |

---

## 8. Tareas pendientes (al 12/06/2026)

- [x] Integrar el dataset de Roboflow (`data/mate-termo/`, 155 imgs) al repo
- [x] Corregir `data/data.yaml` (orden de clases `0=factura, 1=mate, 2=termo`)
- [x] Dejar los notebooks listos para Run-All en Colab (sanity check incluido)
- [x] Actualizar `README.md` con el dataset real (155 imgs, Roboflow)
- [x] Borrador de presentación en `presentacion/presentacion.md`
- [ ] **CRÍTICO** — Entrenar en Colab (Run All del train notebook) y validar
      que el sanity check visual y las métricas dan bien (meta: mAP50 ≥ 0.70)
- [ ] Correr inferencia sobre los 3 videos y verificar la salida visualmente
- [ ] Capturar material para las slides (loss, PR, confusion, frames) y
      completar los `[INSERTAR: ...]` de `presentacion/presentacion.md`
- [ ] Hacer commit final y push antes del **15/06/2026**

---

## 9. Notas para agentes de IA

- **NO agregar dependencias nuevas** sin discutirlo antes — mantener
  `requirements.txt` mínimo.
- **NO cambiar la arquitectura del pipeline** (3 modelos en paralelo) sin
  justificación documentada.
- Si encontrás un bug, **explicar la causa raíz** antes de proponer fix.
- Si entrenás un modelo, **documentar las métricas resultantes** en la
  sección 5.1 (o crear un `EXPERIMENTS.md`).
- **No commitear pesos** (`.pt`) ni outputs pesados. El `.gitignore` ya
  los excluye, mantenerlo así.
- Si el usuario está apurado (deadline 15/06), priorizar funcionalidad
  sobre perfección.

---

## 10. Contacto / contexto

- **Materia:** Visión por Computadora 2026
- **Institución:** UGR (Universidad Gastón Dachary u otra — verificar)
- **Deadline:** Lunes 15 de junio de 2026
- **Entregables:** ver checklist en [`README.md`](./README.md) y [`TAREAS.md`](./TAREAS.md)
