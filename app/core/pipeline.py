# app/core/pipeline.py
import asyncio
import threading
import time
import cv2

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.model_manager import ModelPool
from app.core.processing import draw_detections


class VideoStreamPipeline:
    """
    封装一个视频流的完整处理流水线。
    每个实例对应一个独立的视频源，并管理其下的工作线程。
    通过向模型池借用/归还模型来执行推理。
    """

    def __init__(self, settings: AppSettings, stream_id: str, video_source: str,
                 output_queue: asyncio.Queue, model_pool: ModelPool):
        self.settings = settings
        self.hailo_settings = settings.hailo
        self.stream_id = stream_id
        self.video_source = video_source
        self.output_queue = output_queue
        self.model_pool = model_pool  # 持有模型池的引用

        self._model = None  # 将要从池中借用的模型
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self):
        """启动流水线的主工作线程。"""
        app_logger.info(f"【流水线 {self.stream_id}】正在启动...")
        self.thread = threading.Thread(target=self.run, name=f"Pipeline-{self.stream_id}")
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """请求停止流水线线程。"""
        if self.stop_event.is_set():
            return
        app_logger.info(f"【流水线 {self.stream_id}】正在请求停止...")
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        app_logger.info(f"✅【流水线 {self.stream_id}】已安全停止。")

    def run(self):
        """流水线的主循环，在单独的线程中执行。"""
        # 1. 从模型池获取一个模型
        self._model = self.model_pool.acquire()
        if not self._model:
            app_logger.error(f"❌【流水线 {self.stream_id}】无法获取模型，线程即将退出。")
            return

        app_logger.info(f"【流水线 {self.stream_id}】成功从池中获取模型。")

        # 2. 打开视频源
        source_for_cv = int(self.video_source) if self.video_source.isdigit() else self.video_source
        cap = cv2.VideoCapture(source_for_cv)
        if not cap.isOpened():
            app_logger.error(f"❌【流水线 {self.stream_id}】无法打开视频源: {self.video_source}")
            self.model_pool.release(self._model)  # 确保归还模型
            return

        last_rec_time = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                app_logger.warning(f"【流水线 {self.stream_id}】视频流结束或读取失败。")
                break

            current_time = time.time()
            if current_time - last_rec_time < self.settings.app.stream_recognition_interval_seconds:
                time.sleep(0.01)  # 避免空转消耗过多CPU
                continue
            last_rec_time = current_time

            # 3. 执行推理
            try:
                detection_result = self._model.predict(frame)

                # 4. 绘制结果
                result_frame = draw_detections(
                    frame,
                    detection_result.results,
                    self.hailo_settings.class_names
                )

                # 5. 编码并推送到Web端队列
                (flag, encodedImage) = cv2.imencode(".jpg", result_frame)
                if flag:
                    try:
                        self.output_queue.put_nowait(encodedImage.tobytes())
                    except asyncio.QueueFull:
                        # 队列满了，丢弃当前帧，防止阻塞
                        pass
            except Exception as e:
                app_logger.error(f"【流水线 {self.stream_id}】处理帧时发生错误: {e}", exc_info=False)

        # 6. 循环结束，清理资源
        cap.release()
        if self._model:
            self.model_pool.release(self._model)
        try:
            # 发送结束信号给Web端
            self.output_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        app_logger.info(f"【流水线 {self.stream_id}】工作线程已结束，资源已释放。")