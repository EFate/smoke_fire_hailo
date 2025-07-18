# app/core/pipeline.py
import asyncio
import queue
import threading
import time
from typing import List

import cv2
import onnxruntime as ort

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.model_manager import model_manager
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

        # --- 从 ModelManager 获取共享的模型资源 ---
        self.session: ort.InferenceSession = model_manager.get_session()  #
        self.input_shape: tuple[int, int] = model_manager.get_input_shape()  #
        self.input_name: str = self.session.get_inputs()[0].name  #

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
            if self.preprocess_queue.full():  #
                try:
                    # 队列已满，非阻塞地取出一个元素（最旧的帧）以丢弃
                    self.preprocess_queue.get_nowait()  #
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

                input_tensor, scale, dw, dh = preprocess(frame, self.input_shape)

                # 将处理所需的所有数据打包放入下一队列
                self.inference_queue.put((frame, input_tensor, scale, dw, dh))

            except queue.Empty:
                continue
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

                original_frame, input_tensor, scale, dw, dh = data

                # 执行阻塞的 session.run()
                outputs = self.session.run(None, {self.input_name: input_tensor})  #

                # 将原始帧和推理结果一同传递
                self.postprocess_queue.put((original_frame, outputs[0], scale, dw, dh))

            except queue.Empty:
                continue
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
                if current_time - last_rec_time < self.settings.app.stream_recognition_interval_seconds:  #
                    continue
                last_rec_time = current_time

                original_frame, outputs, scale, dw, dh = data

                boxes, scores, class_ids = postprocess(
                    outputs, scale, dw, dh,
                    self.yolo_settings.confidence_threshold,
                    self.yolo_settings.iou_threshold
                )  #
                result_frame = draw_detections(
                    original_frame, boxes, scores, class_ids, self.yolo_settings.class_names
                )  #

                # 编码并放入最终的异步输出队列
                (flag, encodedImage) = cv2.imencode(".jpg", result_frame)  #
                if flag:
                    try:
                        self.output_queue.put_nowait(encodedImage.tobytes())  #
                    except asyncio.QueueFull:
                        pass

            except queue.Empty:
                continue
        app_logger.info(f"【后处理线程 {self.stream_id}】已停止。")