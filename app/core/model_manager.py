# app/core/model_manager.py
import asyncio
import queue
import threading
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from degirum_tools.inference_models import Model
from degirum_tools.model_loader import load_model, ModelLoaderError
from degirum_tools.profiling import latency_decorator

from app.core.process_utils import get_all_degirum_worker_pids, cleanup_degirum_workers_by_pids

from app.cfg.config import AppSettings, get_app_settings
from app.cfg.logging import app_logger


class ModelPool:
    """
    一个单例类，负责管理 DeGirum 模型的池化。
    这个模型池将管理固定数量的 DeGirum 模型实例，以支持并发推理，
    并确保在应用关闭时能够强制清理所有 DeGirum 相关进程，释放硬件资源。
    """
    _instance = None
    _model_queue: Optional[queue.Queue] = None
    _loaded_models: List[Model] = [] # 用于跟踪所有加载的模型实例
    _model_input_shape: Optional[Tuple[int, int]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelPool, cls).__new__(cls)
            cls._instance.settings: AppSettings = get_app_settings()
        return cls._instance

    async def load_models(self, pool_size: int = 1):
        """
        异步加载 DeGirum 模型并初始化模型池。
        这个方法将在应用启动时调用，加载指定数量的模型实例。
        """
        if self._model_queue is not None and not self._model_queue.empty():
            app_logger.info("模型池已加载，无需重复操作。")
            return

        model_path = str(self.settings.yolo.model_path) # DeGirum 模型是目录
        app_logger.info(f"准备加载 DeGirum 模型: {model_path}，池大小: {pool_size}")

        if not self.settings.yolo.model_path.is_dir():
            error_msg = f"DeGirum 模型目录未找到: {model_path}。请确保模型目录存在于指定路径。"
            app_logger.critical(error_msg)
            raise FileNotFoundError(error_msg)

        self._model_queue = queue.Queue(maxsize=pool_size)

        load_tasks = [
            asyncio.to_thread(self._load_single_model, model_path, i + 1)
            for i in range(pool_size)
        ]

        try:
            # 等待所有模型加载完成
            loaded_models = await asyncio.gather(*load_tasks, return_exceptions=True)

            for model_or_exception in loaded_models:
                if isinstance(model_or_exception, Exception):
                    raise model_or_exception # 重新抛出第一个遇到的异常

                model, idx = model_or_exception
                self._model_queue.put(model)
                self._loaded_models.append(model)
                app_logger.info(f"✅ DeGirum 模型实例 {idx} 加载成功。")

            # 预热并获取模型输入形状（只需对第一个模型执行）
            if self._loaded_models:
                first_model = self._loaded_models[0]
                # Degirum模型通常在predict时自动处理尺寸，但我们可以获取其期望的输入尺寸
                # 例如：(C, H, W) 或 (B, C, H, W)
                # YOLO模型通常是 (B, C, H, W)
                input_shape_full = first_model.input_shape
                if len(input_shape_full) == 4: # (B, C, H, W)
                    self._model_input_shape = (input_shape_full[2], input_shape_full[3]) # (H, W)
                elif len(input_shape_full) == 3: # (C, H, W)
                     self._model_input_shape = (input_shape_full[1], input_shape_full[2]) # (H, W)
                else:
                    raise RuntimeError(f"未知模型输入形状格式: {input_shape_full}")

                # DeGirum 模型通常不需要显式预热，第一次推理会进行 JIT 编译和优化
                # 但是为了统一，我们仍然模拟一个预热过程，确保其内部优化完成
                app_logger.info("正在预热 DeGirum 模型...")
                dummy_input = np.random.rand(self._model_input_shape[0], self._model_input_shape[1], 3).astype(np.uint8)
                await asyncio.to_thread(first_model.predict, dummy_input) # 使用真实predict来预热
                app_logger.info(f"✅ DeGirum 模型池加载并预热成功。模型输入尺寸: {self._model_input_shape[0]}x{self._model_input_shape[1]}")
            else:
                raise RuntimeError("未能加载任何 DeGirum 模型实例。")

        except ModelLoaderError as e:
            app_logger.critical(f"❌ DeGirum 模型加载失败: {e}")
            app_logger.critical("请确保您的 Hailo 设备已正确连接并配置，且模型路径正确无误。")
            raise RuntimeError(f"DeGirum 模型加载失败: {e}") from e
        except Exception as e:
            app_logger.exception(f"❌ DeGirum 模型池初始化时发生未知错误: {e}")
            raise RuntimeError(f"DeGirum 模型池初始化失败: {e}") from e

    def _load_single_model(self, model_path: str, idx: int) -> Tuple[Model, int]:
        """同步执行加载单个 DeGirum 模型的操作。"""
        app_logger.info(f"正在加载 DeGirum 模型实例 {idx}...")
        model = load_model(model_path)
        app_logger.info(f"DeGirum 模型实例 {idx} 加载完成。")
        return model, idx

    @latency_decorator(logger=app_logger, component_name="ModelPool.acquire")
    def acquire(self) -> Model:
        """
        从模型池中获取一个 DeGirum 模型实例。
        如果池中没有可用模型，会阻塞直到有模型可用。
        """
        if self._model_queue is None:
            raise RuntimeError("模型池尚未初始化。请确保在应用启动时调用了 `load_models()`。")
        app_logger.debug("尝试从模型池获取模型...")
        model = self._model_queue.get()
        app_logger.debug("成功获取模型。")
        return model

    @latency_decorator(logger=app_logger, component_name="ModelPool.release")
    def release(self, model: Model):
        """
        将一个 DeGirum 模型实例归还到模型池中。
        """
        if self._model_queue is None:
            app_logger.warning("模型池尚未初始化，无法归还模型。")
            return
        self._model_queue.put(model)
        app_logger.debug("模型已归还到模型池。")

    def get_input_shape(self) -> Tuple[int, int]:
        """获取模型的输入高度和宽度。"""
        if self._model_input_shape is None:
            raise RuntimeError("模型输入尺寸尚未初始化。")
        return self._model_input_shape

    def get_input_name(self) -> str:
        """获取模型输入层的名称（对于 DeGirum 通常不直接需要）。"""
        return "input" # 返回一个默认值

    async def dispose(self):
        """
        在应用关闭时释放所有 DeGirum 模型资源，并强制清理所有 DeGirum 后台进程。
        这个方法将确保没有僵尸进程残留，彻底释放 Hailo 硬件资源。
        """
        if self._loaded_models:
            app_logger.info("正在释放 DeGirum 模型资源并清理相关进程...")
            # 清空队列，确保所有借出的模型都被回收
            while not self._model_queue.empty():
                try:
                    self._model_queue.get_nowait()
                except queue.Empty:
                    pass # should not happen if we are draining

            # Explicitly delete model objects to trigger __del__ or resource release
            # DeGirum模型在Python对象被GC时会自动释放底层资源，但为了确保，我们显式清除引用
            self._loaded_models.clear()
            self._model_queue = None
            self._model_input_shape = None

            # ❗【关键】使用 process_utils 清理所有 DeGirum 工作进程
            pids_to_kill = get_all_degirum_worker_pids()
            cleanup_degirum_workers_by_pids(pids_to_kill, app_logger)

            app_logger.info("✅ DeGirum 模型资源和相关进程已成功释放/清理。")
        else:
            app_logger.info("没有加载的 DeGirum 模型需要释放。")


# 创建单例实例，供应用全局使用
model_pool = ModelPool()

# 封装为 FastAPI 的启动和关闭事件函数，使 main.py 中的调用更语义化。
async def load_degirum_models_on_startup():
    settings = get_app_settings()
    # 默认加载的模型实例数量可以根据您 Hailo 芯片的实际并发推理能力和应用需求调整
    # 例如，Hailo-8 通常可以支持 1-3 路 640x640 YOLOv8 的实时推理
    # 这里我们设置为3，您可以根据需要增加
    await model_pool.load_models(pool_size=settings.app.hailo_model_pool_size)

async def dispose_degirum_models_on_shutdown():
    await model_pool.dispose()