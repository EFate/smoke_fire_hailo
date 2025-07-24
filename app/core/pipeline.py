# app/core/pipeline.py
import asyncio
import queue
import threading
import time
from typing import List, Tuple, Any  # 导入 Any 用于 Degirum 模型结果

import cv2
import numpy as np

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger

from app.core.model_manager import model_pool

from degirum_tools.inference_models import Model
from app.core.processing import preprocess, postprocess, draw_detections


class VideoStreamPipeline:
    """
    封装一个视频流的完整四级处理流水线。
    每个实例对应一个独立的视频源，并管理其下的所有处理线程。
    """

    def __init__(self, settings: AppSettings, stream_id: str, video_source: str, output_queue: asyncio.Queue):
        self.settings = settings
        self.yolo_settings = settings.yolo
        self.stream_id = stream_id
        self.video_source = video_source
        self.output_queue = output_queue  # (Asyncio) 用于将最终结果发送给Web端


        self.model: Optional[Model] = None  # DeGirum 模型实例
        self.input_shape: tuple[int, int] = model_pool.get_input_shape()  # 仍然从 model_pool 获取输入形状

        # --- 为流水线创建三个同步缓冲区队列 ---
        self.preprocess_queue = queue.Queue(maxsize=32)
        self.inference_queue = queue.Queue(maxsize=32)
        self.postprocess_queue = queue.Queue(maxsize=32)

        # --- 线程管理 ---
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []
        self.cap = None

    def start(self):
        """启动所有流水线工作线程。"""
        app_logger.info(f"【流水线 {self.stream_id}】正在启动...")
        try:
            # ❗【新增】从模型池获取模型
            self.model = model_pool.acquire()
            if self.model is None:
                raise RuntimeError("无法从模型池获取 DeGirum 模型实例。")
            app_logger.info(f"【流水线 {self.stream_id}】已从模型池获取模型。")

            self._start_threads()
            app_logger.info(f"✅【流水线 {self.stream_id}】所有线程已启动。")
        except Exception as e:
            app_logger.error(f"❌【流水线 {self.stream_id}】启动失败: {e}", exc_info=True)
            self.stop()  # 确保在启动失败时也能清理资源

    def stop(self):
        """停止所有流水线工作线程并释放资源。"""
        if self.stop_event.is_set():
            return
        app_logger.info(f"【流水线 {self.stream_id}】正在停止...")
        self.stop_event.set()

        # 等待所有线程结束
        for t in self.threads:
            t.join(timeout=2.0)

        # 释放摄像头资源
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # 清空所有队列，以防有线程因队列阻塞而无法退出
        for q in [self.preprocess_queue, self.inference_queue, self.postprocess_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # 向Web端发送最终的结束信号
        try:
            self.output_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass


        if self.model:
            model_pool.release(self.model)
            self.model = None  # 清除引用
            app_logger.info(f"【流水线 {self.stream_id}】已将模型归还到池中。")

        app_logger.info(f"✅【流水线 {self.stream_id}】已安全停止。")

    def _start_threads(self):
        """创建并启动流水线的四个核心线程。"""
        source_for_cv = int(self.video_source) if self.video_source.isdigit() else self.video_source
        self.cap = cv2.VideoCapture(source_for_cv)  #
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.video_source}")

        self.threads = [
            threading.Thread(target=self._reader_thread, name=f"Reader-{self.stream_id}"),
            threading.Thread(target=self._preprocessor_thread, name=f"Preprocessor-{self.stream_id}"),
            threading.Thread(target=self._inference_thread, name=f"Inference-{self.stream_id}"),
            threading.Thread(target=self._postprocessor_thread, name=f"Postprocessor-{self.stream_id}")
        ]
        for t in self.threads:
            t.start()

    # ==============================================================================
    # 四个线程的执行函数
    # ==============================================================================

    def _reader_thread(self):
        """[线程1: 读帧] 以最快速度读取帧，并自动丢弃旧帧以保证处理最新帧。"""
        app_logger.info(f"【读帧线程 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                app_logger.warning(f"【读帧线程 {self.stream_id}】无法读取帧，流结束。")
                break

            # --- START: 保证处理最新帧的关键逻辑 ---
            # 策略：始终保持队列最新。如果队列满了，就丢弃最旧的帧，放入新帧。
            if self.preprocess_queue.full():
                try:
                    # 队列已满，非阻塞地取出一个元素（最旧的帧）以丢弃
                    self.preprocess_queue.get_nowait()
                except queue.Empty:
                    # 在多线程的竞争条件下，可能刚判断完full()队列就被消费了，这是正常情况。
                    pass

            # 将新帧放入队列。因为我们已确保有空间，所以这里可以安全地使用put。
            self.preprocess_queue.put(frame)
            time.sleep(0.01)

        self.preprocess_queue.put(None)  # 发送结束信号
        app_logger.info(f"【读帧线程 {self.stream_id}】已停止。")

    def _preprocessor_thread(self):
        """[线程2: 预处理] 从预处理队列取帧，处理后放入推理队列。"""
        app_logger.info(f"【预处理线程 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            try:
                frame = self.preprocess_queue.get(timeout=1.0)
                if frame is None:
                    self.inference_queue.put(None)  # 传递结束信号
                    break

                # DeGirum模型通常直接接收原始图像，其内部处理预处理
                # 但为了兼容processing.py的后处理逻辑，我们仍计算scale, dw, dh
                # 实际传递给 Degirum model.predict() 的是原始图像 frame
                # 这里仍然调用preprocess来获取scale, dw, dh，但input_tensor可能不直接用于degirum模型
                input_tensor, scale, dw, dh = preprocess(frame, self.input_shape)  #

                # 将原始帧和预处理参数打包放入下一队列
                self.inference_queue.put((frame, scale, dw, dh))

            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【预处理线程 {self.stream_id}】发生错误: {e}", exc_info=True)
                break  # 发生错误时退出线程
        app_logger.info(f"【预处理线程 {self.stream_id}】已停止。")

    def _inference_thread(self):
        """[线程3: 推理] 从推理队列取数据，执行模型推理，结果放入后处理队列。"""
        app_logger.info(f"【推理线程 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            try:
                data = self.inference_queue.get(timeout=1.0)
                if data is None:
                    self.postprocess_queue.put(None)  # 传递结束信号
                    break

                original_frame, scale, dw, dh = data  # ❗【修改】不再接收 input_tensor

                if self.model is None:
                    app_logger.error(f"【推理线程 {self.stream_id}】模型未加载，无法执行推理。")
                    break

                # 执行 DeGirum 模型的推理
                # DeGirum model.predict() 直接接受 OpenCV 格式的 np.ndarray 图像
                inference_results = self.model.predict(original_frame)
                # print(f"DeGirum Inference Results: {inference_results}") # Debugging

                # 将原始帧、DeGirum 结果和预处理参数一同传递
                self.postprocess_queue.put((original_frame, inference_results, scale, dw, dh))

            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【推理线程 {self.stream_id}】发生错误: {e}", exc_info=True)
                break  # 发生错误时退出线程
        app_logger.info(f"【推理线程 {self.stream_id}】已停止。")

    def _postprocessor_thread(self):
        """[线程4: 后处理/显示] 从后处理队列取数据，完成最终处理并放入输出队列。"""
        app_logger.info(f"【后处理线程 {self.stream_id}】启动。")
        last_rec_time = 0
        while not self.stop_event.is_set():
            try:
                data = self.postprocess_queue.get(timeout=1.0)
                if data is None:
                    break  # 收到结束信号，退出循环

                current_time = time.time()
                # 控制后处理的频率，避免过快刷新
                if current_time - last_rec_time < self.settings.app.stream_recognition_interval_seconds:
                    # 如果未到间隔时间，将数据重新放回队列头部（如果队列允许，否则丢弃）
                    # 更好的做法是直接跳过处理，不放回队列
                    continue
                last_rec_time = current_time

                # 接收 DeGirum 的推理结果
                original_frame, inference_results, scale, dw, dh = data

                # 适配 postprocess 函数以处理 DeGirum 的结果
                # DeGirum 模型的 predict 方法通常返回一个包含 'results' 键的字典列表，
                # 每个结果字典包含 'bbox', 'score', 'label' 等。
                # 需要将 DeGirum 的结果转换为 postprocess 函数期望的 (boxes, scores, class_ids) 格式

                boxes_list = []
                scores_list = []
                class_ids_list = []

                # Degirum模型返回的results通常是一个列表，每个元素代表一个检测到的对象
                # 每个对象是一个字典，包含'bbox'、'score'、'label'
                if inference_results and hasattr(inference_results, 'results') and inference_results.results:
                    for det_obj in inference_results.results:  # 假设 Degirum 返回的 results 对象有 .results 属性
                        # Degirum的bbox已经是 [x1, y1, x2, y2] 形式
                        bbox = det_obj.get('bbox')
                        score = det_obj.get('score')
                        label = det_obj.get('label')

                        if bbox and score is not None and label is not None:
                            boxes_list.append(bbox)
                            scores_list.append(score)
                            # 将类别名称映射为ID
                            try:
                                class_id = self.yolo_settings.class_names.index(label)
                                class_ids_list.append(class_id)
                            except ValueError:
                                app_logger.warning(f"未知类别名称: {label}")
                                continue

                boxes = np.array(boxes_list) if boxes_list else np.array([])
                scores = np.array(scores_list) if scores_list else np.array([])
                class_ids = np.array(class_ids_list) if class_ids_list else np.array([])

                # DeGirum 模型直接输出原始图像坐标系的 bbox。
                # `postprocess` 函数现在只做 NMS 过滤
                boxes_final, scores_final, class_ids_final = postprocess(
                    boxes, scores, class_ids,  # ❗【修改】只传递已经处理过的检测结果
                    self.yolo_settings.confidence_threshold,
                    self.yolo_settings.iou_threshold,
                    # DeGirum 模型输出已经是原始图像尺寸，不再需要这些参数进行反变换，但函数签名需要匹配
                    scale=1.0, dw=0, dh=0  # 占位符，实际不再用于反变换
                )

                result_frame = draw_detections(
                    original_frame, boxes_final, scores_final, class_ids_final, self.yolo_settings.class_names
                )

                # 编码并放入最终的异步输出队列
                (flag, encodedImage) = cv2.imencode(".jpg", result_frame)
                if flag:
                    try:
                        self.output_queue.put_nowait(encodedImage.tobytes())
                    except asyncio.QueueFull:
                        pass  # 队列满时丢弃帧

            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【后处理线程 {self.stream_id}】发生错误: {e}", exc_info=True)
                break  # 发生错误时退出线程
        app_logger.info(f"【后处理线程 {self.stream_id}】已停止。")