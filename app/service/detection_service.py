# app/service/detection_service.py
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import HTTPException, status

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.pipeline import VideoStreamPipeline
from app.schema.detection_schema import ActiveStreamInfo, StreamStartRequest


class DetectionService:
    """
    封装核心业务逻辑的服务类。
    职责：
    1. 作为 VideoStreamPipeline 实例的工厂和管理者。
    2. 维护活动流的状态，处理API请求。
    3. 协调应用的生命周期（启动、停止、清理）。
    """

    def __init__(self, settings: AppSettings):
        app_logger.info("正在初始化 DetectionService (重构版)...")
        self.settings = settings
        self.active_streams: Dict[str, VideoStreamPipeline] = {}
        self.stream_lock = asyncio.Lock()  #

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        """启动一个新的视频流处理任务。"""
        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes

        async with self.stream_lock:
            if stream_id in self.active_streams:
                raise HTTPException(status_code=409, detail="Stream ID conflict.")

            # 1. 创建Web端消费的异步队列
            frame_queue = asyncio.Queue(maxsize=self.settings.app.stream_max_queue_size)

            # 2. 实例化完整的处理流水线
            pipeline = VideoStreamPipeline(
                settings=self.settings,
                stream_id=stream_id,
                video_source=req.source,
                output_queue=frame_queue
            )

            # 3. 在线程池中启动流水线（这是一个阻塞操作）
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, pipeline.start)

            # 4. 记录新流的信息
            self.active_streams[stream_id] = pipeline
            started_at = datetime.now()  #
            expires_at = None if lifetime == -1 else started_at + timedelta(minutes=lifetime)  #
            stream_info = ActiveStreamInfo(stream_id=stream_id, source=req.source, started_at=started_at,
                                           expires_at=expires_at, lifetime_minutes=lifetime)

            # (注意) 这里我们将 info 存在 pipeline 对象上，方便清理时访问
            setattr(pipeline, 'info', stream_info)

            app_logger.info(f"🚀 视频流处理流水线已启动: ID={stream_id}, 源={req.source}")
            return stream_info

    async def stop_stream(self, stream_id: str) -> bool:
        """停止一个指定的视频流。"""
        async with self.stream_lock:
            pipeline = self.active_streams.pop(stream_id, None)
            if not pipeline:
                app_logger.warning(f"尝试停止一个不存在或已被停止的流: {stream_id}")
                return False

        # 在线程池中停止流水线（这是一个阻塞操作）
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pipeline.stop)

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
                frame_bytes = await frame_queue.get()
                if frame_bytes is None:
                    app_logger.info(f"接收到流 {stream_id} 的终止信号，正常关闭推送。")
                    break
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                frame_queue.task_done()
        except asyncio.CancelledError:
            app_logger.info(f"客户端从流 {stream_id} 断开，将自动停止该流。")
            await self.stop_stream(stream_id)
            raise

    async def get_all_active_streams_info(self) -> List[ActiveStreamInfo]:
        """获取所有当前活动流的信息列表。"""
        async with self.stream_lock:
            return [getattr(stream, 'info') for stream in self.active_streams.values() if hasattr(stream, 'info')]

    async def cleanup_expired_streams(self):
        """[后台任务] - 定期检查并清理所有已过期的视频流。"""
        while True:
            await asyncio.sleep(self.settings.app.stream_cleanup_interval_seconds)
            now = datetime.now()
            expired_stream_ids = []
            async with self.stream_lock:
                for stream_id, pipeline in self.active_streams.items():
                    info = getattr(pipeline, 'info', None)
                    if info and info.expires_at and now >= info.expires_at:
                        expired_stream_ids.append(stream_id)

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