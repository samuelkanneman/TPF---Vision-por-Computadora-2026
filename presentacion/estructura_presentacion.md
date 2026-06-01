# 📊 Estructura de la Presentación - Trabajo Final Visión por Computadora 2026

Duración sugerida: **10-15 minutos** + preguntas.

---

## Slide 1 — Portada
- Título del proyecto
- Nombres del grupo (hasta 5 integrantes)
- Fecha
- Materia / Carrera

## Slide 2 — Introducción / Contexto
- Qué es YOLO y por qué es relevante
- Objetivo del trabajo
- Desafíos planteados

## Slide 3 — Objetivos
- Entrenar YOLO26 con clases personalizadas
- Integrar detección, segmentación y pose
- Demostrar el sistema sobre video

## Slide 4 — Dataset
- Clases elegidas (mate, termo, factura — Set A)
- Por qué elegimos esas clases
- Cantidad de imágenes por clase
- Herramienta de etiquetado usada (Roboflow / LabelImg / otra)
- Ejemplos visuales de las imágenes

## Slide 5 — Configuración del Entrenamiento
- Modelo base: `yolo26n.pt` (transfer learning)
- Épocas, batch size, imgsz
- Optimizador y learning rate
- Augmentations aplicadas

## Slide 6 — Métricas de Entrenamiento
- Gráficas de loss (train/val)
- mAP50, mAP50-95, precision, recall
- Matriz de confusión
- Curvas PR

## Slide 7 — Resultados del Modelo Custom
- Ejemplos de predicciones correctas
- Ejemplos de errores / falsos positivos
- Discusión: qué funcionó bien y qué no

## Slide 8 — Arquitectura del Pipeline de Inferencia
- Diagrama de bloques:
  - Video → 3 modelos en paralelo → Fusión → Frame anotado → Video de salida
- Modelos usados:
  - YOLO26 Segmentation (COCO)
  - YOLO26 Pose (personas)
  - YOLO26 Detection custom (best.pt)

## Slide 9 — Lógica de Exclusión
- Por qué excluimos `person` de la segmentación
- Por qué excluimos las clases custom de la segmentación
- Cómo se distribuyen las detecciones entre los 3 modelos

## Slide 10 — Demostración: Video Procesado
- Reproducir (o capturar frames del) video final procesado
- Mostrar:
  - Segmentaciones de objetos COCO (sin personas ni custom)
  - Skeletons de personas
  - Bounding boxes de mate/termo/factura con confianza

## Slide 11 — Ejemplos de Frames
- 3-4 frames destacados del video procesado
- Comparativa antes/después

## Slide 12 — Dificultades Encontradas
- Recolección de datos (variabilidad, iluminación, ángulos)
- Confusión entre clases similares
- Performance / FPS del pipeline
- Resolución

## Slide 13 — Conclusiones
- Se cumplieron los objetivos
- Aprendajes del trabajo en equipo
- Posibles mejoras / líneas futuras (más clases, más datos, tracking entre frames, etc.)

## Slide 14 — Preguntas
- "Muchas gracias. ¿Preguntas?"

---

## 🎨 Tips para las slides

- Usar **pocos textos, muchas imágenes y capturas**.
- Incluir **capturas de Roboflow** mostrando el dataset etiquetado.
- Mostrar **gráficas generadas por Ultralytics** (están en `runs/detect/train/`).
- Capturar **frames del video final** con OpenCV para slides estáticas.
- Si el video es muy pesado para la presentación, convertirlo a GIF corto o usar thumbnails.
