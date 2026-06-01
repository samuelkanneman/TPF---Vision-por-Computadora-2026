# 🎓 Trabajo Final - Visión por Computadora 2026

**Detección, Segmentación y Estimación de Pose utilizando YOLO26**

Proyecto integrador que aplica modelos de la familia YOLO para detectar clases personalizadas, segmentar objetos COCO y estimar la pose humana, todo en un único pipeline de procesamiento de video.

---

## 📁 Estructura del proyecto

```
Trabajo Final/
├── data/                          # Dataset custom (no se sube a git si es grande)
│   ├── images/
│   │   ├── train/                 # ≥40 imágenes por clase
│   │   └── val/                   # ≥10 imágenes por clase
│   ├── labels/
│   │   ├── train/                 # Etiquetas en formato YOLO
│   │   └── val/
│   └── data.yaml                  # Configuración del dataset
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

## 🎯 Clases personalizadas (Set A por defecto)

| ID | Clase   | Descripción                            |
|----|---------|----------------------------------------|
| 0  | mate    | Mate (recipiente para yerba mate)      |
| 1  | termo   | Termo para agua caliente               |
| 2  | factura | Factura / pan dulce de panadería       |

> Si querés usar otras clases (Set B: herramientas, Set C: deportivo, Set D: EPP), modificá la sección `names` de `data/data.yaml` y re-etiquetá las imágenes.

---

## 🚀 Quick start

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Armar el dataset

1. Recolectar **≥50 imágenes por clase** (fotos propias, datasets públicos o descarga web).
2. Etiquetar con [Roboflow](https://roboflow.com), [CVAT](https://cvat.org), [LabelImg](https://github.com/tzutalin/labelImg) o MakeSense.ai en formato YOLO.
3. Dividir en `train` (80%) y `val` (20%).
4. Colocar imágenes y labels en `data/images/{train,val}/` y `data/labels/{train,val}/` respectivamente.
5. Verificar que `data/data.yaml` apunta a las rutas correctas.

### 3. Entrenar el modelo

Abrir y ejecutar `training/train_notebook.ipynb` celda por celda. Al finalizar, se generará `runs/detect/train/weights/best.pt`.

### 4. Procesar un video

1. Copiar 2-3 videos cortos (~20 segundos) a `videos/input/`.
2. Abrir y ejecutar `inference/inference_notebook.ipynb`.
3. El video procesado quedará en `videos/output/`.

---

## 🧪 Modelo y configuración usada

- **Modelo base**: `yolo26n.pt` (transfer learning)
  - Si no existe esa versión exacta, reemplazar por la disponible (p.ej. `yolo11n.pt`, `yolo12n.pt`).
- **Épocas**: 80
- **Batch size**: 16
- **Image size**: 640
- **Optimizador**: AdamW
- **Augmentations**: Activadas (default Ultralytics)

> Ajustar estos valores según VRAM disponible y tiempo de entrenamiento.

---

## 📦 Entregables

- [x] Notebooks (training + inference)
- [x] Scripts (`utils.py`)
- [x] Dataset etiquetado (`data/`)
- [x] `data/data.yaml`
- [x] `best.pt` (pesos entrenados)
- [ ] Video original
- [ ] Video procesado final
- [ ] Presentación (siguiendo `presentacion/estructura_presentacion.md`)

---

## 📅 Fechas importantes

- **Entrega notebooks**: Lunes 15 de junio
- **Presentación**: Primer día hábil de la última semana de cursada

---

## ⚠️ Notas

- **"YOLO26"** no es un nombre estándar de la familia YOLO al momento de la fecha. La librería `ultralytics` descarga automáticamente la última versión disponible. Si tu curso указал una versión específica, actualizá los checkpoints (`yolo26n.pt`, `yolo26n-seg.pt`, `yolo26n-pose.pt`) en los notebooks.
- **Clases excluidas en segmentación COCO**: `person` y los nombres de las clases custom (mate, termo, factura). Las personas se detectan con el modelo de **Pose** y las clases custom con el modelo **Detection** entrenado.
