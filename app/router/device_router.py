# app/router/device_router.py
from fastapi import APIRouter, HTTPException, status, Request, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, TypeVar, Generic
import asyncio
from app.cfg.logging import app_logger  # 导入日志模块

# --- 通用 API 响应模型 (在此文件中定义，以自包含所有设备接口相关Schema) ---
T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """
    标准化的API响应体结构。
    所有API端点都应返回此结构，以便前端或客户端能够统一处理。
    """
    code: int = Field(0, description="响应状态码，0表示成功，其它非零值表示特定的业务失败")
    msg: str = Field("Success", description="响应消息，提供操作结果的文本描述")
    data: Optional[T] = Field(None, description="实际的响应数据。其具体结构由泛型 T 决定。")


# --- Hailo 设备信息 Schema ---
class DeviceInfo(BaseModel):
    """单个 Hailo 设备的信息。"""
    device_id: str = Field(..., description="Hailo 设备的唯一标识符")
    average_power_watts: Optional[float] = Field(None, description="当前平均功耗（瓦特），如果不可用则为 None")
    chip_temperature_celsius: Optional[float] = Field(None, description="芯片温度（摄氏度），如果不可用则为 None")


class GetAllDevicesResponseData(BaseModel):
    """获取所有 Hailo 设备信息的响应数据。"""
    device_count: int = Field(..., description="检测到的 Hailo 设备总数")
    devices: List[DeviceInfo] = Field([], description="所有 Hailo 设备的详细信息列表")


router = APIRouter()


# --- Hailo 设备信息接口 ---
@router.get(
    "/device",
    response_model=ApiResponse[GetAllDevicesResponseData],
    summary="获取 Hailo 设备信息",
    description="获取所有连接的 Hailo 设备的状态，包括功耗和温度。",
    tags=["Hailo设备"]  # 更改为更具体的标签
)
async def get_hailo_devices():
    """
    返回所有 Hailo 设备的详细信息。
    """
    device_list: List[DeviceInfo] = []
    try:
        # 延迟导入 hailo_platform，避免在非 Hailo 环境下服务启动失败
        from hailo_platform import Device, HailoRTException

        # 使用 asyncio.to_thread 运行阻塞的 Hailo 设备扫描和查询操作
        def _scan_and_get_info_sync():
            """同步函数，用于在单独的线程中执行 Hailo 设备扫描和信息获取。"""
            device_infos = Device.scan()
            targets = [Device(di) for di in device_infos]

            results = []
            for di, target in zip(device_infos, targets):
                power = None
                temp = None
                try:
                    # 尝试获取功率和温度
                    # 用户提供的输出已经包含警告，表明不建议频繁开关测量
                    # 因此，我们只调用 get_power_measurement 和 get_chip_temperature
                    power = target.control.get_power_measurement().average_value
                    temp = target.control.get_chip_temperature().ts0_temperature

                except HailoRTException as e:
                    app_logger.warning(f"无法获取 Hailo 设备 {di} 的功耗或温度: {e}. 可能设备繁忙或不可用。")
                except Exception as e:
                    app_logger.error(f"获取 Hailo 设备 {di} 功耗或温度时发生未知错误: {e}")

                results.append(
                    DeviceInfo(
                        device_id=str(di),
                        average_power_watts=power,
                        chip_temperature_celsius=temp
                    )
                )
            return results

        device_list = await asyncio.to_thread(_scan_and_get_info_sync)

    except ImportError:
        app_logger.error("Hailo 平台库未安装或环境未激活。无法获取设备信息。")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Hailo 平台库不可用，无法查询设备信息。")
    except Exception as e:
        app_logger.error(f"获取 Hailo 设备信息时发生未预期的错误: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取设备信息失败: {e}")

    return ApiResponse(
        data=GetAllDevicesResponseData(
            device_count=len(device_list),
            devices=device_list
        )
    )