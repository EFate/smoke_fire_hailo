# app/core/model_manager.py
import queue
import gc
import degirum as dg
from degirum.predict import DeGirumModel
from typing import Optional, Tuple

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.process_utils import get_all_degirum_worker_pids, cleanup_degirum_workers_by_pids


class ModelPool:
    """
    负责管理 DeGirum 模型实例池的单例类。
    职责:
    1. 应用启动时，根据池大小预加载多个模型实例。
    2. 为并发任务提供模型的“借用”(acquire)和“归还”(release)机制。
    3. 应用关闭时，调用强制清理逻辑，确保所有DeGirum后台进程被终止，硬件资源被完全释放。
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelPool, cls).__new__(cls)
        return cls._instance

    def __init__(self, settings: AppSettings, pool_size: int):
        if hasattr(self, '_initialized') and self._initialized:
            return

        app_logger.info(f"正在初始化 DeGirum 模型池，大小为: {pool_size}...")
        self.settings = settings
        self.pool_size = pool_size
        self._pool = queue.Queue(maxsize=pool_size)

        try:
            # 记录启动前的 DeGirum 进程，以便只清理本次启动创建的
            self._initial_pids = get_all_degirum_worker_pids()
            app_logger.info(f"启动前检测到 {len(self._initial_pids)} 个残留 DeGirum 进程。")

            for i in range(pool_size):
                app_logger.info(f"正在加载模型实例 {i + 1}/{pool_size}...")
                model = self._create_degirum_model()
                self._pool.put(model)
            app_logger.info("✅ DeGirum 模型池已成功加载并填充。")
        except Exception as e:
            app_logger.critical(f"❌ 初始化 DeGirum 模型池失败: {e}", exc_info=True)
            app_logger.critical("请检查模型名称是否正确，以及 Hailo 设备是否连接并正常工作。")
            self.dispose()  # 尝试清理已创建的资源
            raise RuntimeError(f"模型池初始化失败: {e}") from e

        self._initialized = True

    def _create_degirum_model(self) -> DeGirumModel:
        """使用配置中的信息加载单个 DeGirum 模型实例。"""
        model = dg.load_model(
            model_name=self.settings.hailo.detection_model_name,
            inference_host_address=dg.LOCAL,
            zoo_url=self.settings.hailo.zoo_url,
            image_backend='opencv',
            confidence_threshold=self.settings.hailo.confidence_threshold,
            nms_threshold=self.settings.hailo.iou_threshold
        )
        if not model:
            raise ConnectionError("dg.load_model 返回 None，无法加载模型。")
        return model

    def acquire(self, timeout: float = 2.0) -> Optional[DeGirumModel]:
        """从池中获取一个模型实例。如果池为空，将等待指定时间。"""
        try:
            return self._pool.get(timeout=timeout)
        except queue.Empty:
            app_logger.error(f"在 {timeout}s 内未能从池中获取可用模型，服务可能过载。")
            return None

    def release(self, model: DeGirumModel):
        """将一个模型实例归还到池中。"""
        try:
            self._pool.put_nowait(model)
        except queue.Full:
            # 如果池满了，说明归还逻辑可能有问题，直接丢弃该模型实例
            app_logger.warning("尝试将模型归还到已满的池中，此实例将被丢弃。")
            del model

    def dispose(self):
        """
        应用关闭时调用的核心清理函数。
        强制终止所有由本次运行创建的 DeGirum 工作进程。
        """
        app_logger.warning("正在释放 DeGirum 模型池资源并清理后台进程...")

        # 1. 识别所有当前正在运行的 DeGirum 进程
        all_current_pids = get_all_degirum_worker_pids()

        # 2. 计算出由本次应用实例创建的新进程
        pids_to_kill = all_current_pids - self._initial_pids

        # 3. 强制终止这些新进程
        cleanup_degirum_workers_by_pids(pids_to_kill, app_logger)

        # 4. 清理 Python 端的对象
        while not self._pool.empty():
            try:
                model = self._pool.get_nowait()
                del model
            except queue.Empty:
                break
        gc.collect()
        app_logger.info("✅ DeGirum 模型池资源已清理。")