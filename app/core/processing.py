# app/core/processing.py
from typing import Tuple, List
import cv2
import numpy as np


def preprocess(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, int, int]:
    """对输入图像进行预处理，以满足YOLO模型的输入要求。"""
    img_h, img_w, _ = image.shape
    input_h, input_w = input_shape
    scale = min(input_w / img_w, input_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    dw, dh = (input_w - new_w) // 2, (input_h - new_h) // 2
    padded_img[dh:dh + new_h, dw:dw + new_w, :] = resized_img
    img_rgb = padded_img[:, :, ::-1]
    img_transposed = np.transpose(img_rgb, (2, 0, 1))
    input_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32) / 255.0
    return input_tensor, scale, dw, dh

def postprocess(
    output: np.ndarray,
    scale: float,
    dw: int,
    dh: int,
    conf_threshold: float,
    iou_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对模型输出进行后处理，将原始输出转换为易于使用的边界框、分数和类别ID。"""
    predictions = np.squeeze(output).T
    scores = np.max(predictions[:, 4:], axis=1)
    mask = scores > conf_threshold
    predictions = predictions[mask, :]
    scores = scores[mask]
    if not predictions.shape[0]:
        return np.array([]), np.array([]), np.array([])
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes_xywh = predictions[:, :4]
    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy -= np.array([dw, dh, dw, dh])
    boxes_xyxy /= scale
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes_xyxy[indices], scores[indices], class_ids[indices]
    return np.array([]), np.array([]), np.array([])

def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: List[str]
):
    """在图像上绘制检测框、类别和置信度，用于可视化。"""
    colors = {"smoke": (200, 200, 200), "fire": (0, 0, 255)}
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names[class_id]
        color = colors.get(class_name, (0, 255, 0))
        label = f"{class_name}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image