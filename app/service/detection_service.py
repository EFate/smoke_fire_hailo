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

from app.core.model_manager import model_pool


class DetectionService:
    """
    封装核心业务逻辑的服务类，持有对 ModelPool 的引用。
    职责：
    1. 作为 VideoStreamPipeline 实例的工厂和管理者。
    2. 维护活动流的状态，处理API请求。
    3. 协调应用的生命周期（启动、停止、清理）。
    """

    def __init__(self, settings: AppSettings):
        app_logger.info("正在初始化 DetectionService (适配 Hailo)...")  # 更新日志信息
        self.settings = settings
        self.model_pool = model_pool  # 直接引用 ModelPool 单例
        self.active_streams: Dict[str, VideoStreamPipeline] = {}
        self.stream_lock = asyncio.Lock()

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        """启动一个新的视频流处理任务。"""
        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes

        app_logger.info(f"准备为流 {stream_id} 启动一个新的视频流处理流水线 (它将从模型池中获取模型)...")

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
            try:
                # pipeline.start() 内部会 acquire 模型
                await loop.run_in_executor(None, pipeline.start)
            except Exception as e:
                app_logger.error(f"启动视频流 {stream_id} 失败: {e}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"无法启动视频流：{e}")

            # 4. 记录新流的信息
            self.active_streams[stream_id] = pipeline
            started_at = datetime.now()
            expires_at = None if lifetime == -1 else started_at + timedelta(minutes=lifetime)
            stream_info = ActiveStreamInfo(stream_id=stream_id, source=req.source, started_at=started_at,
                                           expires_at=expires_at, lifetime_minutes=lifetime)

            setattr(pipeline, 'info', stream_info)  # 记录 info 到 pipeline 对象上

            app_logger.info(f"🚀 视频流处理流水线已启动: ID={stream_id}, 源={req.source}")
            return stream_info

    async def stop_stream(self, stream_id: str) -> bool:
        """停止一个指定的视频流。"""
        async with self.stream_lock:
            pipeline = self.active_streams.pop(stream_id, None)
            if not pipeline:
                app_logger.warning(f"尝试停止一个不存在或已被停止的流: {stream_id}")
                return False

        # 在线程池中停止流水线（这是一个阻塞操作），pipeline.stop() 内部会 release 模型
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
                # 阻塞获取帧，直到有帧可用或收到终止信号
                frame_bytes = await frame_queue.get()
                if frame_bytes is None:
                    app_logger.info(f"接收到流 {stream_id} 的终止信号，正常关闭推送。")
                    break
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                frame_queue.task_done()
        except asyncio.CancelledError:
            app_logger.info(f"客户端从流 {stream_id} 断开，将自动停止该流。")
            await self.stop_stream(stream_id)  # 客户端断开时自动停止流
            raise
        except Exception as e:
            app_logger.error(f"获取流 {stream_id} 馈送时发生错误: {e}", exc_info=True)
            await self.stop_stream(stream_id)  # 发生错误时停止流
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="流处理异常，已停止。")

    async def get_all_active_streams_info(self) -> List[ActiveStreamInfo]:
        """获取所有当前活动流的信息列表。"""
        async with self.stream_lock:
            # 过滤掉没有 info 属性的（理论上不应该有）或者线程已死的管道
            active_pipelines = [p for p in self.active_streams.values() if
                                hasattr(p, 'info') and (p.threads and p.threads[0].is_alive())]
            return [getattr(p, 'info') for p in active_pipelines]

    async def cleanup_expired_streams(self):
        """[后台任务] - 定期检查并清理所有已过期的视频流。"""
        while True:
            await asyncio.sleep(self.settings.app.stream_cleanup_interval_seconds)
            now = datetime.now()
            expired_stream_ids = []
            async with self.stream_lock:
                # 遍历 active_streams 的副本，以避免在迭代时修改字典
                for stream_id, pipeline in list(self.active_streams.items()):
                    info = getattr(pipeline, 'info', None)
                    # 检查流是否过期，或者管道线程是否已意外停止
                    if (info and info.expires_at and now >= info.expires_at) or \
                            (pipeline.threads and not pipeline.threads[
                                0].is_alive() and not pipeline.stop_event.is_set()):
                        app_logger.warning(f"检测到流 {stream_id} 过期或已意外停止，准备清理。")
                        expired_stream_ids.append(stream_id)

            if expired_stream_ids:
                app_logger.info(f"🗑️ 发现 {len(expired_stream_ids)} 个过期/异常视频流，正在清理: {expired_stream_ids}")
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