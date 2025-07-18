# app/core/model_manager.py
import asyncio
from typing import Optional, Tuple, List
import onnxruntime as ort
import numpy as np

from app.cfg.config import AppSettings, get_app_settings
from app.cfg.logging import app_logger


class ModelManager:
    """
    一个单例类，负责管理 ONNX 推理会话的整个生命周期。
    这是保证模型资源被高效、安全地使用的核心。
    主要职责:
    1. 模型的加载与初始化。
    2. 智能选择执行提供者 (Execution Provider)，优先使用GPU (CUDA)。
    3. 模型预热 (Warm-up)，减少首次请求的延迟。
    4. 资源的统一释放。
    """
    _instance = None
    _session: Optional[ort.InferenceSession] = None
    _model_input_shape: Optional[Tuple[int, int]] = None

    def __new__(cls):
        # 实现单例模式，确保整个应用中只有一个 ModelManager 实例。
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.settings: AppSettings = get_app_settings()
        return cls._instance

    async def load_model(self):
        """
        异步加载 ONNX 模型。这是整个服务启动过程中的一个关键且耗时的步骤。
        它被设计为在应用启动时（lifespan中）调用一次。
        """
        if self._session is not None:
            app_logger.info("模型已加载，无需重复操作。")
            return

        model_path = str(self.settings.yolo.model_path.resolve())
        app_logger.info(f"准备加载 ONNX 模型: {model_path}")

        if not self.settings.yolo.model_path.exists():
            error_msg = f"模型文件未找到: {model_path}。请确保模型文件存在于指定路径。"
            app_logger.critical(error_msg)
            raise FileNotFoundError(error_msg)

        # --- 智能检测并选择执行提供者 ---
        # 这是本模块最核心的逻辑之一，它带来了巨大的灵活性和健壮性。
        # 应用可以无缝地在有GPU和无GPU的环境中部署，无需修改任何代码。
        preferred_providers = self.settings.yolo.providers
        available_providers = ort.get_available_providers()
        final_providers: List[str] = []

        # 按配置文件中的优先级尝试使用提供者
        for provider in preferred_providers:
            if provider in available_providers:
                app_logger.info(f"✅ 检测到可用的 '{provider}'。将尝试使用它进行推理。")
                final_providers.append(provider)
                break  # 找到第一个可用的，就停止搜索

        if not final_providers:
            app_logger.warning(f"⚠️ 配置文件中请求的提供者 {preferred_providers} 都不可用。")
            app_logger.warning(f"   - 系统实际可用: {available_providers}")
            app_logger.warning("   - 将自动回退到 CPU 执行。")
            final_providers.append('CPUExecutionProvider')

        app_logger.info(f"最终使用的 ONNX Runtime providers: {final_providers}")

        try:
            loop = asyncio.get_running_loop()
            # 将创建会话这个同步、阻塞的操作放入线程池中执行，避免阻塞asyncio事件循环。
            self._session = await loop.run_in_executor(
                None, self._create_inference_session, model_path, final_providers
            )

            # 获取并存储模型输入尺寸，后续的预处理会用到
            input_meta = self._session.get_inputs()[0]
            height, width = input_meta.shape[2], input_meta.shape[3]
            self._model_input_shape = (height, width)

            # --- 模型预热 (Warm-up) ---
            # 第一次运行推理通常会比较慢，因为它需要进行一些初始化和内存分配。
            # 在启动时用一个虚拟输入运行一次推理，可以“预热”模型，确保第一个真实请求的响应速度。
            app_logger.info("正在预热模型...")
            await loop.run_in_executor(
                None,
                self._warmup_model,
                (1, 3, height, width),  # 创建一个符合模型输入的虚拟数据
                self._session.get_inputs()[0].name
            )
            app_logger.info(f"✅ ONNX 模型加载并预热成功。模型输入尺寸: {height}x{width}")

        except Exception as e:
            # 详尽的错误处理和用户指引是优秀软件的标志。
            error_message = str(e)
            if "CUDA" in error_message or "GPU" in error_message:
                app_logger.critical("❌ ONNX Runtime 加载 CUDA 执行提供者失败！这通常是GPU环境配置问题。")
                guide_message = (
                    "GPU调用失败，请严格检查您的环境：\n"
                    "1. [驱动检查] 在终端运行 `nvidia-smi` 命令，确保NVIDIA驱动程序正常工作。\n"
                    "2. [版本匹配] 确认安装的 `onnxruntime-gpu` 版本与系统中的 CUDA Toolkit 和 cuDNN 版本完全匹配。\n"
                    "3. [依赖安装] 确保 CUDA Toolkit 和 cuDNN 已正确安装并配置了所有相关的环境变量 (如 PATH, LD_LIBRARY_PATH)。\n"
                    "4. [临时方案] 如果问题无法解决，可以修改配置文件 `default.yaml`，将 `providers` 改为 `['CPUExecutionProvider']` 以强制使用CPU模式启动服务。"
                )
                app_logger.error(guide_message)
                # 抛出致命错误，阻止服务启动，因为这是无法自动恢复的配置问题。
                raise RuntimeError("GPU环境配置错误，无法启动服务。请检查日志获取详细指引。") from e
            else:
                app_logger.exception(f"❌ ONNX 模型加载时发生未知错误: {e}")
                raise RuntimeError(f"ONNX 模型初始化失败: {e}")

    def _create_inference_session(self, model_path: str, providers: List[str]) -> ort.InferenceSession:
        """同步执行的推理会话创建函数，用于在 executor 中调用。"""
        return ort.InferenceSession(model_path, providers=providers)

    def _warmup_model(self, shape: tuple, input_name: str):
        """同步执行的模型预热函数。"""
        dummy_input = np.random.rand(*shape).astype(np.float32)
        self._session.run(None, {input_name: dummy_input})

    def get_session(self) -> ort.InferenceSession:
        """获取已加载的推理会话实例。在服务运行时被高频调用。"""
        if self._session is None:
            raise RuntimeError("模型尚未加载。请确保在应用启动时调用了 `load_model()`。")
        return self._session

    def get_input_shape(self) -> Tuple[int, int]:
        """获取模型的输入高度和宽度。"""
        if self._model_input_shape is None:
            raise RuntimeError("模型输入尺寸尚未初始化。")
        return self._model_input_shape

    async def release_resources(self):
        """在应用关闭时释放模型资源。"""
        if self._session is not None:
            app_logger.info("正在释放 ONNX 模型资源...")
            # 在 onnxruntime 中，将 session 设置为 None 就足以让垃圾回收器回收相关内存。
            # ort没有显式的 close() 或 del_session() 方法。
            self._session = None
            self._model_input_shape = None
            app_logger.info("✅ 模型资源已成功释放。")


# 创建单例实例，供应用全局（主要是DetectionService和lifespan函数）使用
model_manager = ModelManager()


# 封装为 FastAPI 的启动和关闭事件函数，使 main.py 中的调用更语义化。
async def load_models_on_startup():
    await model_manager.load_model()


async def release_models_on_shutdown():
    await model_manager.release_resources()