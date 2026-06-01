"""
utils.py - Funciones auxiliares para el pipeline de inferencia.

Incluye:
  - Dibujo de bounding boxes de clases custom
  - Dibujo de mascaras de segmentacion (con exclusion de clases)
  - Dibujo de keypoints y skeleton de pose estimation
  - Overlay de informacion (FPS, conteos)
  - Generador deterministico de colores por clase
"""

import cv2
import numpy as np


# ---------------------------------------------------------------
# Paleta de colores (BGR) - determinada por class_id
# ---------------------------------------------------------------
def get_color(class_id: int) -> tuple:
    """Devuelve un color BGR estable para un class_id."""
    np.random.seed(int(class_id) + 7)
    return tuple(int(c) for c in np.random.randint(60, 255, 3))


# ---------------------------------------------------------------
# Bounding boxes de clases custom
# ---------------------------------------------------------------
def draw_custom_boxes(image, result, model_names, conf_threshold=0.4):
    """
    Dibuja bounding boxes de clases custom con label y confianza.

    Args:
        image: np.ndarray BGR (frame original).
        result: ultralytics.Results (de un modelo Detection entrenado).
        model_names: dict {class_id: name} del modelo.
        conf_threshold: confianza minima a dibujar.

    Returns:
        np.ndarray BGR con las cajas dibujadas.
    """
    annotated = image.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    for box, cls, conf in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
    ):
        if float(conf) < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls)
        name = model_names[cls_id]
        confidence = float(conf)
        color = get_color(cls_id + 100)  # offset para no chocar con seg

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - th - 10)),
            (x1 + tw, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(th, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return annotated


# ---------------------------------------------------------------
# Segmentacion (con exclusion de clases)
# ---------------------------------------------------------------
def draw_segmentation_masks(image, result, model_names, exclude_classes=None, conf_threshold=0.4):
    """
    Dibuja mascaras de segmentacion, excluyendo las clases indicadas.

    Args:
        image: np.ndarray BGR.
        result: ultralytics.Results (de un modelo Segmentation).
        model_names: dict {class_id: name}.
        exclude_classes: set/iterable de nombres a NO dibujar (ej: {'person', 'mate'}).
        conf_threshold: confianza minima.

    Returns:
        (annotated, n_drawn) - frame con mascaras y cantidad dibujadas.
    """
    if exclude_classes is None:
        exclude_classes = set()
    exclude_classes = set(exclude_classes)

    annotated = image.copy()
    n_drawn = 0

    if result.masks is None or result.boxes is None:
        return annotated, 0

    h, w = image.shape[:2]
    masks_data = result.masks.data.cpu().numpy()           # (N, Hm, Wm)
    classes = result.boxes.cls.cpu().numpy().astype(int)   # (N,)
    confs = result.boxes.conf.cpu().numpy()

    for i in range(len(masks_data)):
        if float(confs[i]) < conf_threshold:
            continue

        name = model_names[int(classes[i])]
        if name in exclude_classes:
            continue

        # Resize de la mascara al tamano del frame
        mask = cv2.resize(masks_data[i], (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bool = mask > 0.5

        color = get_color(int(classes[i]))
        overlay = annotated.copy()
        overlay[mask_bool] = color
        annotated = cv2.addWeighted(annotated, 0.55, overlay, 0.45, 0)

        # Contorno de la mascara
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, color, 2)

        # Label chico al costado del primer contorno
        if contours:
            x, y, ww, hh = cv2.boundingRect(contours[0])
            cv2.putText(
                annotated,
                name,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        n_drawn += 1

    return annotated, n_drawn


# ---------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------
# Pares de keypoints (COCO 17) que forman el esqueleto
SKELETON_PAIRS = [
    (16, 14), (14, 12), (17, 15), (15, 13),
    (12, 13), (6, 12), (7, 13), (6, 7),
    (6, 8), (7, 9), (8, 10), (9, 11),
    (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]


def draw_pose(image, result, kp_threshold=0.5):
    """
    Dibuja keypoints y skeleton para todas las personas detectadas.

    Args:
        image: np.ndarray BGR.
        result: ultralytics.Results (de un modelo Pose).
        kp_threshold: confianza minima por keypoint.

    Returns:
        (annotated, n_persons)
    """
    annotated = image.copy()
    n_persons = 0

    if result.keypoints is None or result.keypoints.data is None:
        return annotated, 0

    persons = result.keypoints.data.cpu().numpy()   # (N, 17, 3)
    n_persons = len(persons)

    for person in persons:
        # Dibujar lineas del esqueleto
        for (a, b) in SKELETON_PAIRS:
            if a - 1 < len(person) and b - 1 < len(person):
                xa, ya, ca = person[a - 1]
                xb, yb, cb = person[b - 1]
                if ca > kp_threshold and cb > kp_threshold:
                    cv2.line(
                        annotated,
                        (int(xa), int(ya)),
                        (int(xb), int(yb)),
                        (0, 255, 255),
                        2,
                    )
        # Dibujar los keypoints como circulos
        for kp in person:
            x, y, c = kp
            if c > kp_threshold:
                cv2.circle(annotated, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.circle(annotated, (int(x), int(y)), 4, (255, 255, 255), 1)

    return annotated, n_persons


# ---------------------------------------------------------------
# Overlay de informacion (FPS y conteos)
# ---------------------------------------------------------------
def add_info_overlay(image, fps, custom_count=0, seg_count=0, person_count=0):
    """
    Dibuja un panel con FPS y conteos de objetos/personas detectados.
    """
    h, w = image.shape[:2]
    panel_w = 320
    panel_h = 30 + 25 * 3 + 10

    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    y = 35
    cv2.putText(
        image, f"FPS: {fps:.1f}", (20, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
    y += 25
    cv2.putText(
        image, f"Personas (pose): {person_count}", (20, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    y += 22
    cv2.putText(
        image, f"Clases custom: {custom_count}", (20, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    y += 22
    cv2.putText(
        image, f"Segmentos COCO: {seg_count}", (20, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )

    return image


# ---------------------------------------------------------------
# Funcion todo-en-uno: aplica los 3 modelos y devuelve el frame
# ---------------------------------------------------------------
def annotate_frame(frame, seg_result, pose_result, custom_result,
                   seg_model_names, pose_model_names, custom_model_names,
                   custom_class_names, conf_threshold=0.4):
    """
    Combina segmentacion COCO (sin person/custom), pose, y bboxes custom
    en un unico frame anotado.
    """
    canvas = frame.copy()

    # 1) Segmentacion COCO - excluir 'person' y clases custom
    canvas, n_seg = draw_segmentation_masks(
        canvas, seg_result, seg_model_names,
        exclude_classes={"person"} | set(custom_class_names),
        conf_threshold=conf_threshold,
    )

    # 2) Pose estimation (personas)
    canvas, n_persons = draw_pose(canvas, pose_result)

    # 3) Bounding boxes de clases custom
    canvas = draw_custom_boxes(
        canvas, custom_result, custom_model_names,
        conf_threshold=conf_threshold,
    )
    n_custom = 0
    if custom_result.boxes is not None:
        n_custom = len(custom_result.boxes)

    return canvas, {
        "n_seg": n_seg,
        "n_persons": n_persons,
        "n_custom": n_custom,
    }
