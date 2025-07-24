# app/core/processing.py
from typing import Tuple, List
import cv2
import numpy as np


def preprocess(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, int, int]:
    """
    对输入图像进行预处理，以满足YOLO模型的输入要求。
    对于DeGirum模型，其predict方法通常直接接收原始图像并内部处理，
    此函数返回的 input_tensor 可能不再直接用于DeGirum模型的predict方法。
    但返回的 scale, dw, dh 参数在 postprocess 中仍可能用于坐标转换。
    """
    img_h, img_w, _ = image.shape
    input_h, input_w = input_shape
    scale = min(input_w / img_w, input_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    dw, dh = (input_w - new_w) // 2, (input_h - new_h) // 2
    padded_img[dh:dh + new_h, dw:dw + new_w, :] = resized_img
    img_rgb = padded_img[:, :, ::-1]  # 转换为 RGB
    img_transposed = np.transpose(img_rgb, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32) / 255.0  # Add batch dim and normalize
    return input_tensor, scale, dw, dh


def postprocess(
        # DeGirum 模型直接给出 bbox, score, class_id 列表。
        boxes_degirum: np.ndarray,
        scores_degirum: np.ndarray,
        class_ids_degirum: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        scale: float,
        dw: int,
        dh: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 DeGirum 模型输出进行后处理，主要执行 NMS。
    DeGirum 模型通常已完成了大部分后处理，直接给出在原始图像坐标系下的结果。
    """
    predictions_count = boxes_degirum.shape[0] if boxes_degirum.ndim > 1 else 0

    if not predictions_count:
        return np.array([]), np.array([]), np.array([])

    # 过滤置信度阈值
    mask = scores_degirum > conf_threshold
    boxes_filtered = boxes_degirum[mask]
    scores_filtered = scores_degirum[mask]
    class_ids_filtered = class_ids_degirum[mask]

    if not boxes_filtered.shape[0]:
        return np.array([]), np.array([]), np.array([])

    # 执行 NMS
    # cv2.dnn.NMSBoxes 期望的 boxes 是 [x1, y1, x2, y2]
    indices = cv2.dnn.NMSBoxes(
        boxes_filtered.tolist(), scores_filtered.tolist(), conf_threshold, iou_threshold
    )

    if len(indices) > 0:
        indices = indices.flatten()
        return boxes_filtered[indices], scores_filtered[indices], class_ids_filtered[indices]

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

        # 确保 class_id 在有效范围内
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = "unknown"
            # app_logger.warning(f"检测到未知 class_id: {class_id}，请检查模型输出和 class_names 配置。") # 导入 app_logger 需要从 logging.py
            # 暂时移除，避免循环引用或不必要的导入
            pass

        color = colors.get(class_name, (0, 255, 0))  # 默认绿色
        label = f"{class_name}: {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image