# ✅ Tareas Pendientes — Trabajo Final Visión por Computadora 2026

Roadmap operativo desde hoy (1 de junio) hasta la entrega del **lunes 15 de junio**.

> **Faltan 14 días.** El plan ya está hecho y la estructura del repo está en GitHub. Ahora hay que ejecutar.

---

## 🗓 Cronograma macro

| Sem. | Fechas | Foco | Estado |
|------|--------|------|--------|
| S1 | Lun 2 – Dom 8 | Dataset + setup | ✅ |
| S2 | Lun 9 – Dom 15 | Entrenamiento + video + entrega | ⏳ (al 12/06: falta entrenar, inferir y completar slides) |

---

## 📌 Etapa 1 — Decisiones clave (1-2 días)

> Reunión inicial del grupo. Dejar todo cerrado por escrito.

- [x] **Confirmar 2-3 clases personalizadas** → Set A: factura, mate, termo (orden 0/1/2 del export de Roboflow)
- [x] **Verificar disponibilidad de GPU** → Google Colab (no hay GPU local)
- [x] **Decidir herramienta de etiquetado** → Roboflow
- [x] **Asignar roles** del equipo → Samuel + Fede
- [x] **Confirmar versión de YOLO** → `yolo26n` (disponible en `ultralytics >= 8.4` desde enero 2026)

### 👥 Roles sugeridos (5 personas)

| # | Rol | Responsable | Tareas principales |
|---|-----|-------------|---------------------|
| 1 | Data Lead | _____ | Recolección, limpieza y etiquetado |
| 2 | Trainer | _____ | Entrenamiento, hiperparámetros, métricas |
| 3 | Vision Engineer | _____ | Pipeline de video + integración de 3 modelos |
| 4 | QA / Docs | _____ | Validación, README, métricas para slides |
| 5 | Presenter | _____ | Presentación + selección/grabación de videos |

---

## 📌 Etapa 2 — Dataset (3-4 días)

> Esta es la etapa más larga. La calidad del dataset define el techo del modelo.

- [x] **Recolectar ≥50 imágenes por clase** → 155 imágenes totales (instancias: 172 factura, 55 mate, 35 termo)
- [x] **Verificar que las clases NO estén en COCO** → factura/mate/termo no están en COCO ✓
- [x] **Etiquetar cada imagen** → Roboflow, formato polígono YOLO (ultralytics lo convierte a bboxes para detect)
- [x] **Split 80/20 train/val** → 124 train / 31 valid
- [x] **Integrar al repo** → `data/mate-termo/` (estructura Roboflow, se commitea)
- [x] **Actualizar `data/data.yaml`** → corregido: `0: factura, 1: mate, 2: termo` (orden del export de Roboflow)
- [ ] **Sanity check**: correr la celda 3.1 del train notebook (dibuja labels sobre imágenes) antes de entrenar

### 🛠 Herramientas de etiquetado

| Herramienta | Pros | Contras |
|-------------|------|---------|
| **Roboflow** ⭐ | Cloud, exporta a YOLO directo, augmentations, gratis hasta 10k imgs | Requiere cuenta |
| **LabelImg** | Local, simple, open source | Sin augmentations, manual |
| **CVAT** | Online, colaborativo | Setup inicial más complejo |
| **MakeSense.ai** | Web, sin cuenta | Sin autoguardado de progreso |

---

## 📌 Etapa 3 — Entrenamiento (2-3 días)

> Archivo: `training/train_notebook.ipynb`

- [ ] **Verificar entorno**: `pip install -r requirements.txt`
- [ ] **Verificar GPU** (celda 2 del notebook)
- [ ] **Verificar dataset** (celda 3) — confirmar que cuenta las imágenes y labels correctamente
- [ ] **Ajustar hiperparámetros** en celda 5 según VRAM disponible:
  - `epochs`: 50-100 (recomendado 80)
  - `batch`: 8-32 (16 si tenés 8GB+ VRAM)
  - `imgsz`: 640 (default)
  - `optimizer`: AdamW
- [ ] **Lanzar entrenamiento** (celda 5) — puede tardar 30 min a 4 horas según hardware
- [ ] **Revisar métricas** (celda 6): mAP50, mAP50-95, precision, recall
  - Meta sugerida: **mAP50 ≥ 0.70** para considerar el modelo aceptable
- [ ] **Copiar `best.pt`** a la raíz del proyecto (celda 8)
- [ ] **Si mAP < 0.50**: re-entrenar con más datos, más épocas, o probar `yolo26s.pt`

### 🔄 Iteración si las métricas son bajas

- [ ] Agregar más imágenes al dataset
- [ ] Activar más augmentations (dejar default suele ser suficiente)
- [ ] Probar con `yolo26s.pt` o `yolo26m.pt` (más precisión, más VRAM)
- [ ] Revisar labels mal hechos (clase incorrecta, bbox mal dibujada)

---

## 📌 Etapa 4 — Selección/grabación de videos (1-2 días)

> Archivos: `videos/input/`

- [x] **Elegir/grabar 2-3 videos** → 3 videos en `videos/input/` (video1/2/3.mp4)
- [ ] **Verificar contenido de cada video**:
  - ✅ Al menos 1 persona visible (para pose)
  - ✅ Objetos COCO varios (auto, silla, botella, etc. — para segmentación)
  - ✅ Las 3 clases custom visibles (mate, termo, factura — para bboxes)
- [ ] **Formato**: `.mp4` o `.avi` (codec compatible con OpenCV)
- [ ] **Resolución sugerida**: 720p o 1080p (evitar 4K por performance)
- [x] **Copiar videos** a `videos/input/` (gitignoreados; en Colab llegan por el Drive compartido)

### 💡 Tips para los videos

- **Buena iluminación**: ayuda a la segmentación
- **Distintos ángulos**: valida la robustez del modelo
- **Personas en movimiento**: el esqueleto se ve mejor
- **Fondo no muy cluttered**: facilita la segmentación COCO

---

## 📌 Etapa 5 — Pipeline de inferencia (3-4 días)

> Archivo: `inference/inference_notebook.ipynb` + `inference/utils.py`

- [ ] **Probar el notebook con un video corto primero** (configurar `MAX_FRAMES=100` para debug)
- [ ] **Verificar la salida**:
  - ✅ Segmentaciones COCO se ven (sin `person` ni clases custom)
  - ✅ Skeletons de personas se ven correctamente
  - ✅ Bboxes de clases custom con label + confianza
- [ ] **Ajustar umbral de confianza** si hay demasiado ruido (subir a 0.5-0.6)
- [ ] **Probar con los 2-3 videos** y comparar resultados
- [ ] **Exportar videos finales** a `videos/output/`

### 🐛 Debugging común

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'ultralytics'` | `pip install ultralytics` |
| `FileNotFoundError: best.pt` | Correr primero el notebook de training |
| `VIDEO_PATH not found` | Verificar que el video está en `videos/input/` |
| FPS muy bajo (<5) | Usar `yolo26n` (no `m`), reducir `imgsz` a 320, usar GPU |
| No detecta clases custom | Verificar que `best.pt` está bien entrenado (mAP > 0.5) |
| No detecta personas | Usar `yolo26n-pose.pt`, no `yolo26n.pt` |

---

## 📌 Etapa 6 — Documentación y entrega (2-3 días)

> **Entrega: lunes 15 de junio** · **Presentación: primer día hábil de la última semana**

- [x] **Limpiar notebooks** → outputs limpios, markdown explicativo, listos para Run All en Colab
- [ ] **Actualizar README.md** con los resultados finales (dataset ya actualizado ✓; faltan mAP y FPS post-entrenamiento)
- [ ] **Armar presentación** → borrador con las 14 slides en `presentacion/presentacion.md` ✓; completar los `[INSERTAR: ...]` con métricas y frames
- [ ] **Capturar frames** del video procesado para las slides
- [ ] **Capturar gráficas de Ultralytics** (loss, matriz de confusión, PR) — están en `runs/detect/train/`
- [ ] **Verificar entregables** (checklist abajo)
- [ ] **Hacer commit final** y push a GitHub

### 📦 Checklist de entregables

- [ ] `training/train_notebook.ipynb` (entregable)
- [ ] `inference/inference_notebook.ipynb` (entregable)
- [ ] `inference/utils.py`
- [ ] `data/` con imágenes y labels (o `.zip` aparte si es muy pesado)
- [ ] `data/data.yaml`
- [ ] `best.pt` (pesos entrenados)
- [ ] `videos/input/` — videos originales
- [ ] `videos/output/` — videos procesados finales
- [ ] `presentacion/` — slides en `.pdf` o `.pptx`
- [ ] `README.md` actualizado

---

## 🧪 Comandos útiles

```powershell
# Activar entorno (si usás venv)
cd 'c:\Users\skannema\Downloads\ugr 2026\vision por computadora\Trabajo Final'
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Git: flujo de trabajo después de cada cambio importante
git add .
git commit -m "feat: descripcion del cambio"
git push

# Ver estado del repo
git status
git log --oneline

# Jupyter: lanzar el editor
jupyter notebook
```

---

## 📊 Métricas objetivo

| Métrica | Valor mínimo aceptable | Valor ideal |
|---------|------------------------|-------------|
| mAP50 (clases custom) | ≥ 0.50 | ≥ 0.75 |
| mAP50-95 (clases custom) | ≥ 0.35 | ≥ 0.55 |
| Precision (clases custom) | ≥ 0.60 | ≥ 0.80 |
| Recall (clases custom) | ≥ 0.50 | ≥ 0.70 |
| FPS del pipeline | ≥ 5 | ≥ 15 |

---

## 📞 Decisiones pendientes

Todas resueltas al 12/06:

- [x] ¿Set A, B, C o D de clases? → **Set A** (factura, mate, termo)
- [x] ¿Roboflow o LabelImg? → **Roboflow**
- [x] ¿CPU, GPU local, Colab o Kaggle? → **Google Colab** (Drive: `MyDrive/UGR/PDI_TP2/`)

---

> 💡 **Tip final:** Suban un commit a GitHub **al menos cada 2 días**. Si alguien se atrasa, el resto del grupo tiene el código. Es la red de seguridad más importante para un trabajo grupal.
