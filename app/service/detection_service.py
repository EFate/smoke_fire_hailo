# app/service/detection_service.py
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any

from fastapi import HTTPException, status

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.model_manager import ModelPool
from app.core.pipeline import VideoStreamPipeline
from app.schema.detection_schema import ActiveStreamInfo, StreamStartRequest


class DetectionService:
    """
    封装核心业务逻辑的服务类 (Hailo版)。
    职责：
    1. 管理 VideoStreamPipeline 实例的生命周期。
    2. 注入模型池，使流水线能够共享硬件资源。
    3. 维护活动流的状态，处理API请求。
    """

    def __init__(self, settings: AppSettings, model_pool: ModelPool):
        app_logger.info("正在初始化 DetectionService (Hailo版)...")
        self.settings = settings
        self.model_pool = model_pool
        # active_streams 现在存储 pipeline 对象本身，以便调用其方法
        self.active_streams: Dict[str, VideoStreamPipeline] = {}
        # 存储流的元数据，方便查询
        self.stream_infos: Dict[str, ActiveStreamInfo] = {}
        self.stream_lock = asyncio.Lock()

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        """启动一个新的视频流处理任务。"""
        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes

        async with self.stream_lock:
            # 1. 创建Web端消费的异步队列
            frame_queue = asyncio.Queue(maxsize=self.settings.app.stream_max_queue_size)

            # 2. 实例化流水线，并注入模型池
            pipeline = VideoStreamPipeline(
                settings=self.settings,
                stream_id=stream_id,
                video_source=req.source,
                output_queue=frame_queue,
                model_pool=self.model_pool
            )

            # 3. 启动流水线线程
            pipeline.start()

            # 短暂等待以确认线程是否因为无法获取模型等原因立即失败
            await asyncio.sleep(0.2)
            if not pipeline.thread or not pipeline.thread.is_alive():
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "服务正忙或无法启动处理线程，请稍后再试。")

            # 4. 记录新流的信息
            self.active_streams[stream_id] = pipeline
            started_at = datetime.now()
            expires_at = None if lifetime == -1 else started_at + timedelta(minutes=lifetime)
            stream_info = ActiveStreamInfo(stream_id=stream_id, source=req.source, started_at=started_at,
                                           expires_at=expires_at, lifetime_minutes=lifetime)
            self.stream_infos[stream_id] = stream_info

            app_logger.info(f"🚀 视频流处理线程已启动: ID={stream_id}, 源={req.source}")
            return stream_info

    async def stop_stream(self, stream_id: str) -> bool:
        """停止一个指定的视频流。"""
        async with self.stream_lock:
            pipeline = self.active_streams.pop(stream_id, None)
            _ = self.stream_infos.pop(stream_id, None)
            if not pipeline:
                app_logger.warning(f"尝试停止一个不存在或已被停止的流: {stream_id}")
                return False

        # 在当前协程中请求停止（内部会join线程）
        pipeline.stop()
        app_logger.info(f"✅ 视频流流水线已请求停止: ID={stream_id}")
        return True

    async def get_stream_feed(self, stream_id: str):
        """[前台协程] - 从指定流水线的输出队列获取帧。"""
        async with self.stream_lock:
            pipeline = self.active_streams.get(stream_id)

        if not pipeline:
            app_logger.warning(f"客户端尝试连接一个不存在或已停止的流: {stream_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stream not found or already stopped.")

        frame_queue = pipeline.output_queue
        try:
            while True:
                # 检查流水线线程是否还在运行
                if not pipeline.thread.is_alive() and frame_queue.empty():
                    app_logger.info(f"检测到流 {stream_id} 的后台线程已停止，正常关闭推送。")
                    break

                try:
                    frame_bytes = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                    if frame_bytes is None:
                        break  # 收到结束信号
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    frame_queue.task_done()
                except asyncio.TimeoutError:
                    continue  # 超时是正常的，继续等待下一帧

        except asyncio.CancelledError:
            app_logger.info(f"客户端从流 {stream_id} 断开，将自动停止该流。")
            await self.stop_stream(stream_id)
            raise

    async def get_all_active_streams_info(self) -> List[ActiveStreamInfo]:
        """获取所有当前活动流的信息列表。"""
        async with self.stream_lock:
            # 清理已经意外死掉的流
            dead_stream_ids = [sid for sid, p in self.active_streams.items() if not p.thread.is_alive()]
            for sid in dead_stream_ids:
                self.active_streams.pop(sid, None)
                self.stream_infos.pop(sid, None)
                app_logger.warning(f"检测并清理了一个意外终止的流: {sid}")

            return list(self.stream_infos.values())

    async def cleanup_expired_streams(self):
        """[后台任务] - 定期检查并清理所有已过期的视频流。"""
        while True:
            await asyncio.sleep(self.settings.app.stream_cleanup_interval_seconds)
            now = datetime.now()
            # 创建副本以避免在迭代时修改字典
            streams_to_check = list(self.stream_infos.items())
            expired_stream_ids = [sid for sid, info in streams_to_check if info.expires_at and now >= info.expires_at]

            if expired_stream_ids:
                app_logger.info(f"🗑️ 发现 {len(expired_stream_ids)} 个过期视频流，正在清理: {expired_stream_ids}")
                cleanup_tasks = [self.stop_stream(stream_id) for stream_id in expired_stream_ids]
                await asyncio.gather(*cleanup_tasks)

    async def stop_all_streams(self):
        """在应用关闭时，停止所有活动的视频流。"""
        app_logger.info("应用准备关闭，正在停止所有活动的视频流...")
        async with self.stream_lock:
            all_stream_ids = list(self.active_streams.keys())
        if all_stream_ids:
            stop_tasks = [self.stop_stream(stream_id) for stream_id in all_stream_ids]
            await asyncio.gather(*stop_tasks)
            app_logger.info(f"✅ 所有 {len(all_stream_ids)} 个活动流已清理完毕。")