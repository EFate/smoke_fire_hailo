# app/cfg/config.py
import os
import yaml
from pathlib import Path
from typing import Any, List, Optional
from functools import lru_cache
from pydantic import BaseModel, Field, BeforeValidator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated


# --- 路径定义 ---
# 使用 Path 对象来处理路径，更健壮且跨平台
def get_base_dir() -> Path:
    """计算并返回项目的根目录。"""
    return Path(__file__).resolve().parent.parent.parent


BASE_DIR = get_base_dir()
ENV_FILE = BASE_DIR / ".env"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "app" / "cfg"
DATA_DIR = BASE_DIR / "data"

# --- 自定义类型注解 ---
# 定义一个在验证前将字符串转换为Path对象的类型
FilePath = Annotated[Path, BeforeValidator(lambda v: Path(v) if isinstance(v, str) else v)]


# --- 配置模型定义 (使用 Pydantic 进行验证和类型提示) ---
class AppConfig(BaseModel):
    """应用通用配置"""
    title: str = "高性能烟火检测服务"
    description: str = "基于FastAPI和YOLO Hailo模型构建的实时烟火检测服务" # 描述更新
    version: str = "1.0.0"
    debug: bool = False
    stream_default_lifetime_minutes: int = Field(10, description="视频流默认生命周期（分钟），-1表示永久")
    stream_cleanup_interval_seconds: int = Field(60, description="后台清理过期视频流的运行间隔（秒）")
    stream_recognition_interval_seconds: float = Field(0.1, description="视频流中执行检测的最小间隔（秒），即1/FPS")
    stream_max_queue_size: int = Field(120, description="为视频流提供一个更充裕的缓冲区，以应对客户端网络抖动")
    hailo_model_pool_size: int = Field(3, description="Hailo模型池中的模型实例数量，控制并发推理能力。")


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = Field(False, description="是否开启热重载，仅建议在开发环境开启")


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    file_path: FilePath = LOGS_DIR / "app.log"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5

    @model_validator(mode='after')
    def ensure_log_dir_exists(self) -> 'LoggingConfig':
        """验证后执行，确保日志文件所在的目录存在。"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        return self


class YoloConfig(BaseModel):
    """YOLO模型与推理配置"""
    model_path: FilePath = DATA_DIR / "zoo" / "yolov8n_relu6_fire_smoke--640x640_quant_hailort_hailo8_1"
    class_names: List[str] = ["fire", "smoke"]
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="目标检测置信度阈值")
    iou_threshold: float = Field(0.4, ge=0.0, le=1.0, description="非极大值抑制（NMS）的IOU阈值")

    @model_validator(mode='after')
    def ensure_model_dir_exists(self) -> 'YoloConfig':
        """验证后执行，确保模型文件所在的目录存在。"""
        if not self.model_path.is_dir():
             raise ValueError(f"DeGirum模型目录未找到: {self.model_path}。请确保模型目录存在于指定路径。")
        return self


# --- 主配置类 ---
class AppSettings(BaseSettings):
    """将所有配置模块组合在一起的主配置类。"""
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    yolo: YoloConfig = Field(default_factory=YoloConfig)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",  # 允许通过环境变量覆盖嵌套配置, e.g., APP__DEBUG=true
    )


# --- YAML 配置加载器 ---
class ConfigLoader:
    @staticmethod
    def load_yaml_configs(env: Optional[str] = None) -> dict:
        """加载基础和环境特定的YAML配置文件，并进行深度合并。"""
        current_env = env or os.getenv("APP_ENV", "development").lower()
        config: dict = {}

        default_path = CONFIG_DIR / "default.yaml"
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        env_path = CONFIG_DIR / f"{current_env}.yaml"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                env_config = yaml.safe_load(f) or {}
                config = ConfigLoader._deep_merge_dicts(config, env_config)
        return config

    @staticmethod
    def _deep_merge_dicts(base: dict, updates: dict) -> dict:
        """递归地合并两个字典。"""
        merged = base.copy()
        for key, value in updates.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigLoader._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged


# --- 配置加载接口 (使用 lru_cache 实现单例) ---
@lru_cache(maxsize=1)
def get_app_settings(env_override: Optional[str] = None) -> AppSettings:
    """
    获取全局应用配置的单例实例。
    加载顺序 (后者覆盖前者):
    1. Pydantic模型默认值 -> 2. default.yaml -> 3. [env].yaml -> 4. .env文件 -> 5. 环境变量
    """
    current_env = env_override or os.getenv("APP_ENV", "development")

    # 1. 从YAML文件加载基础配置
    yaml_data = ConfigLoader.load_yaml_configs(current_env)

    # 2. 从环境变量和.env文件加载，并与YAML配置合并
    # pydantic-settings 会自动处理 .env 和环境变量
    # 我们需要将 YAML 的数据作为基础，然后让环境变量覆盖它
    settings_from_env = AppSettings()

    # 将YAML数据验证并转换为Pydantic模型
    base_settings = AppSettings.model_validate(yaml_data)

    # 将模型转回字典，以便与环境变量加载的配置合并
    merged_data = ConfigLoader._deep_merge_dicts(
        base_settings.model_dump(),
        settings_from_env.model_dump(exclude_unset=True)  # exclude_unset=True确保只使用显式设置的环境变量
    )

    return AppSettings.model_validate(merged_data)