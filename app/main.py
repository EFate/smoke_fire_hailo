# app/main.py
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.cfg.config import get_app_settings
from app.cfg.logging import app_logger, setup_logging

from app.core.model_manager import model_pool, load_degirum_models_on_startup, dispose_degirum_models_on_shutdown
from app.router.detection_router import router as detection_router
from app.schema.detection_schema import ApiResponse
from app.service.detection_service import DetectionService

# 在模块加载时执行一次初始化。
# 这是因为 Uvicorn 等服务器可能会在多个工作进程中导入此模块。
settings = get_app_settings()
setup_logging(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器 (Lifespan Manager)。这是 FastAPI 推荐的、用于替代旧的
    `@app.on_event("startup")` 和 `@app.on_event("shutdown")` 的现代方式。
    它能确保在应用启动前完成初始化，在应用关闭后释放资源。
    """
    # --- 启动任务 ---
    app_logger.info("🚀 应用启动中...")

    # 1. ❗【修改】加载 DeGirum 模型池。
    await load_degirum_models_on_startup()

    # 2. 初始化核心服务。
    #    将配置和服务实例附加到 app.state，这是一种在 FastAPI 应用中共享单例对象的标准做法。
    #    所有请求处理函数都可以通过 `request.app.state` 访问到这些实例。
    app.state.settings = settings
    # 将 model_pool 注入到 DetectionService (DetectionService内部直接引用了model_pool单例，这里不再显式注入)
    detection_service = DetectionService(settings=settings)
    app.state.detection_service = detection_service
    app_logger.info("✅ 检测服务 (DetectionService) 初始化完成。")

    # 3. 创建并启动后台任务。
    #    `asyncio.create_task` 会立即开始执行 `cleanup_expired_streams` 协程，
    #    它将在后台独立运行，定期清理过期的视频流。
    cleanup_task = asyncio.create_task(detection_service.cleanup_expired_streams())
    app.state.cleanup_task = cleanup_task
    app_logger.info("✅ 已启动过期视频流的周期性清理任务。")

    app_logger.info("🎉 应用启动成功，准备接收请求！")

    yield  # yield 语句是分界点。Uvicorn 会在此处暂停，开始运行应用主体，处理请求。

    # --- 关闭任务 ---
    # 当应用收到关闭信号（如 Ctrl+C），代码会从 yield 处继续执行。
    app_logger.info("👋 应用关闭中...")

    # 1. 优雅地取消后台清理任务。
    cleanup_task = app.state.cleanup_task
    if cleanup_task and not cleanup_task.done():
        cleanup_task.cancel()
        try:
            # 等待任务响应取消信号，可以确保任务内部的清理逻辑（如果有）能执行。
            await cleanup_task
        except asyncio.CancelledError:
            app_logger.info("✅ 视频流清理任务已成功取消。")

    # 2. 停止所有正在运行的视频流，释放摄像头、文件句柄等硬件资源。
    if hasattr(app.state, 'detection_service'):
        await app.state.detection_service.stop_all_streams()

    # 3. 释放 DeGirum 模型资源和清理相关进程。
    await dispose_degirum_models_on_shutdown()
    app_logger.info("✅ 所有关闭任务已完成。应用已安全退出。")


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例的工厂函数。
    这种模式使得应用更易于测试和配置。
    """
    app = FastAPI(
        lifespan=lifespan,
        title=settings.app.title,
        description=settings.app.description,
        version=settings.app.version,
        debug=settings.app.debug,
        docs_url=None,  # 禁用默认的 /docs，我们将使用自定义路径
        redoc_url=None,
    )

    # --- 全局异常处理器 ---
    # 定义全局异常处理器，可以捕获应用中所有抛出的特定类型异常，
    # 并返回统一格式的JSON响应，这对于构建规范的API至关重要。

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """处理HTTPException，这是FastAPI中业务逻辑错误的常用异常。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ApiResponse(code=exc.status_code, msg=exc.detail).model_dump()
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """处理所有未被捕获的其它异常，防止敏感信息泄露，并确保服务器不会因意外错误而崩溃。"""
        app_logger.exception(f"在处理请求 {request.url} 时发生未处理的异常: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(code=500, msg="服务器内部错误，请联系管理员。").model_dump()
        )

    # --- 挂载路由和静态文件 ---
    # 使用 `include_router` 将在其他文件中定义的路由模块化地包含进来。
    app.include_router(detection_router, prefix="/api/detection", tags=["烟火检测服务"])

    # 挂载静态文件目录，用于提供 Swagger UI 的 JS 和 CSS 文件。
    STATIC_FILES_DIR = Path(__file__).parent / "static"
    if STATIC_FILES_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

    # --- 自定义文档路由 ---
    # 提供一个自定义的 /docs 端点，可以更灵活地控制Swagger UI的外观和行为。
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """提供自定义的 Swagger UI 界面。"""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - API 文档",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
        )

    @app.get("/", tags=["System"], include_in_schema=False)
    async def read_root():
        """根路径，提供欢迎信息和文档链接，方便用户初次访问。"""
        return {"message": f"欢迎使用 {settings.app.title}!", "docs_url": "/docs"}

    return app


# 创建 FastAPI 应用实例，供 uvicorn 在 run.py 中通过 "app.main:app" 引用和启动。
app = create_app()