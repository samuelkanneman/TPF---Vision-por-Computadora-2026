# ✅ Tareas Pendientes — Trabajo Final Visión por Computadora 2026

Roadmap operativo desde hoy (1 de junio) hasta la entrega del **lunes 15 de junio**.

> **Faltan 14 días.** El plan ya está hecho y la estructura del repo está en GitHub. Ahora hay que ejecutar.

---

## 🗓 Cronograma macro

| Sem. | Fechas | Foco | Estado |
|------|--------|------|--------|
| S1 | Lun 2 – Dom 8 | Dataset + setup | ✅ |
| S2 | Lun 9 – Dom 15 | Entrenamiento + video + entrega | ✅ (12/06: entrenado, videos procesados, slides armadas — falta exportar PPTX y entregar) |

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

- [x] **Recolectar ≥50 imágenes por clase** → 155 imágenes totales (instancias en train: 208 factura, 106 mate, 71 termo)
- [x] **Verificar que las clases NO estén en COCO** → factura/mate/termo no están en COCO ✓
- [x] **Etiquetar cada imagen** → Roboflow, formato polígono YOLO (ultralytics lo convierte a bboxes para detect)
- [x] **Split 80/20 train/val** → 124 train / 31 valid
- [x] **Integrar al repo** → `data/mate-termo/` (estructura Roboflow, se commitea)
- [x] **Actualizar `data/data.yaml`** → corregido: `0: factura, 1: mate, 2: termo` (orden del export de Roboflow)
- [x] **Sanity check**: corrido (celda 3.1 del train notebook) — etiquetado y orden de clases verificados ✓

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

- [x] **Verificar entorno** → Colab vía extensión de VS Code (T4)
- [x] **Verificar GPU** → Tesla T4, 15.6 GB
- [x] **Verificar dataset** → 124 train / 31 valid ✓
- [x] **Ajustar hiperparámetros** → `yolo26n`, 150 épocas, batch 16, AdamW, patience 30
- [x] **Lanzar entrenamiento** → 3 corridas (~8 min c/u en T4); final: early stop en época 119
- [x] **Revisar métricas** → **mAP50 0.57** (termo 0.76 ✓, mate 0.66, factura 0.29) — no se alcanzó el 0.70 global; causa analizada y documentada (recall en facturas agrupadas, dataset chico)
- [x] **Copiar `best.pt`** → en la raíz del repo (5.4 MB)
- [x] **Iteración** → se probó `yolo26s`: overfit (0.53); el nano generaliza mejor con este dataset

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
- [x] **Copiar videos** a `videos/input/` (commiteados en el repo, ~6.5 MB — el clon de Colab ya los tiene)

### 💡 Tips para los videos

- **Buena iluminación**: ayuda a la segmentación
- **Distintos ángulos**: valida la robustez del modelo
- **Personas en movimiento**: el esqueleto se ve mejor
- **Fondo no muy cluttered**: facilita la segmentación COCO

---

## 📌 Etapa 5 — Pipeline de inferencia (3-4 días)

> Archivo: `inference/inference_notebook.ipynb` + `inference/utils.py`

- [x] **Probar el notebook** → corrido completo en Colab
- [x] **Verificar la salida**:
  - ✅ Segmentaciones COCO se ven (sin `person`, clases custom, ni donut/cake/cup)
  - ✅ Skeletons de personas se ven correctamente (video1: conductor tomando mate)
  - ✅ Bboxes de clases custom con label + confianza
- [x] **Umbral de confianza** → 0.4 (sin ruido visible)
- [x] **Probar con los 3 videos** → 17.5 / 17.5 / 12.3 FPS en T4
- [x] **Exportar videos finales** → `videos/output/resultado_video{1,2,3}.mp4`

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

- [x] **Limpiar notebooks** → markdown explicativo; se entregan **con los outputs de la corrida final** (para que se vea la ejecución)
- [x] **Actualizar README.md** → resultados finales, historial de experimentos y FPS
- [x] **Armar presentación** → `presentacion/presentacion.md` (15 slides, Marp) con métricas e imágenes reales
- [x] **Capturar frames** → `presentacion/img/frames/` (extraídos de los videos procesados)
- [x] **Capturar gráficas de Ultralytics** → `presentacion/img/` (results, confusion matrix, curvas P/R/PR/F1)
- [x] **Verificar entregables** (checklist abajo)
- [x] **Hacer commit final** y push a GitHub
- [ ] **Exportar presentación a PPTX/PDF** (Marp for VS Code → Export Slide Deck) y ensayarla

### 📦 Checklist de entregables

- [x] `training/train_notebook.ipynb` (entregable, con outputs de la corrida final)
- [x] `inference/inference_notebook.ipynb` (entregable, con outputs de la corrida final)
- [x] `inference/utils.py`
- [x] `data/mate-termo/` con imágenes y labels (commiteado en el repo)
- [x] `data/data.yaml`
- [x] `best.pt` (pesos entrenados, en la raíz)
- [x] `videos/input/` — videos originales (commiteados)
- [x] `videos/output/` — videos procesados finales (commiteados)
- [x] `presentacion/` — fuente Marp + imágenes (exportar a `.pdf`/`.pptx` antes de presentar)
- [x] `README.md` actualizado

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
