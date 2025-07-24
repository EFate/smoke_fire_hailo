# app/service/device_monitor.py
import json
import time
import sys
import os
import threading
from pathlib import Path
from ..cfg.logging import app_logger

try:
    from hailo_platform import Device, HailoRTException
    from hailo_platform.pyhailort.pyhailort import BoardInformation
except ImportError:
    # 在这里不直接退出，而是允许类被导入，在实例化时再处理
    # 这样主应用即使在没有安装库的环境中也能启动（尽管监控功能会失败）
    BoardInformation = None
    HailoRTException = Exception
    Device = None
    print("警告：无法导入 'hilo_platform' 模块。设备监控服务将不可用。")


class DeviceMonitor:
    """
    一个独立的后台服务类，用于周期性监控Hailo设备状态并写入文件。
    """

    def __init__(self, logger = app_logger, interval: int = 5):
        """
        初始化设备监控服务。
        Args:
            logger (Logger): 从主应用传入的日志记录器。
            interval (int): 监控的时间间隔（秒）。
        """
        self.logger = logger
        self.interval = interval
        self._thread = None
        self._stop_event = threading.Event()

        # 从环境变量获取路径，如果未设置，则使用默认值
        default_output_path = "/data/hailo/hailo_device_status.json"
        self.output_path = Path(os.getenv("DEVICE_INFO_FILE_PATH", default_output_path))

    def start(self):
        """在后台线程中启动监控循环。"""
        if self._thread and self._thread.is_alive():
            self.logger.warning("设备监控服务已在运行中。")
            return

        # 检查hailo库是否成功导入
        if Device is None:
            self.logger.error("❌ HailoRT库未加载，无法启动设备监控服务。")
            return

        # 启动前确保输出目录存在
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"监控状态文件将被写入: {self.output_path.absolute()}")
        except OSError as e:
            self.logger.critical(f"❌ 无法创建监控文件输出目录 {self.output_path.parent}: {e}")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        self.logger.info("🚀 Hailo 设备后台监控服务已启动。")

    def stop(self):
        """停止后台监控循环。"""
        if not self._thread or not self._thread.is_alive():
            self.logger.info("设备监控服务未在运行。")
            return

        self.logger.info("👋 正在停止 Hailo 设备后台监控服务...")
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 1)  # 等待线程结束
        if self._thread.is_alive():
            self.logger.warning("监控线程在超时后仍未结束。")
        else:
            self.logger.info("✅ Hailo 设备后台监控服务已成功停止。")
        self._thread = None

    def _fetch_device_metrics(self) -> dict:
        """扫描并获取所有Hailo设备的详细指标。"""
        # (这个函数的内容与您提供的脚本中的 fetch_device_metrics 完全相同)
        device_infos = Device.scan()
        if not device_infos:
            self.logger.warning("未检测到 Hailo 设备。")
            return {"device_count": 0, "devices": []}

        targets = [Device(di) for di in device_infos]
        results = []

        for di, target in zip(device_infos, targets):
            device_data = {"device_id": str(di)}
            try:
                board_info = target.control.identify()
                extended_info = target.control.get_extended_device_information()
                device_data.update({
                    "board_name": board_info.board_name.strip('\x00'),
                    "serial_number": board_info.serial_number.strip('\x00'),
                    "part_number": board_info.part_number.strip('\x00'),
                    "product_name": board_info.product_name.strip('\x00'),
                    "device_architecture": BoardInformation.get_hw_arch_str(board_info.device_architecture),
                    "nn_core_clock_rate_mhz": round(extended_info.neural_network_core_clock_rate / 1_000_000, 1),
                    "boot_source": str(extended_info.boot_source).split('.')[-1],
                })
                current_power = target.control.power_measurement()
                temp_data = target.control.get_chip_temperature()
                device_data.update({
                    "current_power_watts": round(current_power, 3) if current_power is not None else None,
                    "chip_temperature": {
                        "ts0_celsius": temp_data.ts0_temperature,
                        "ts1_celsius": temp_data.ts1_temperature
                    },
                })
            except HailoRTException as e:
                self.logger.warning(f"获取设备 {di} 的部分信息失败: {e}")
            except Exception as e:
                self.logger.error(f"获取设备 {di} 信息时发生未知错误: {e}")

            results.append(device_data)

        # 在返回前释放设备对象
        for target in targets:
            try:
                target.release()
            except HailoRTException:
                pass

        return {"device_count": len(results), "devices": results}

    def _monitoring_loop(self):
        """在后台线程中运行的监控主循环。"""
        while not self._stop_event.is_set():
            metrics_data = self._fetch_device_metrics()
            try:
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"✅ 已将 {metrics_data['device_count']} 个设备的状态更新到 {self.output_path}")
            except IOError as e:
                self.logger.error(f"❌ 写入监控文件失败: {e}")
            except Exception as e:
                self.logger.error(f"处理监控数据时发生未知错误: {e}")

            # 等待指定的时间间隔，或直到停止事件被触发
            self._stop_event.wait(timeout=self.interval)

device_monitor = DeviceMonitor(app_logger, interval=5)