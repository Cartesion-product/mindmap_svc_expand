import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any


class LogManager:
    """
    日志管理器类，使用单例模式实现。
    支持控制台日志和文件日志，可根据配置文件设置日志行为和级别。
    支持日志文件滚动功能。
    支持自定义配置文件路径、名称和logger名称。
    支持选择时间戳格式（unix毫秒时间戳或常规格式）。
    日志格式：[级别] 时间戳 消息
    """

    _instance = None
    _celery_logger = None  # Celery logger 实例

    def __new__(cls, config_dir: Optional[str] = None, config_file: str = "appconfig.json"):
        """
        确保只创建一个LogManager实例（单例模式）
        :param config_dir: 配置文件所在的目录路径，默认为None（当前目录）
        :param config_file: 配置文件名称，默认为"appconfig.json"
        """
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_dir: Optional[str] = None, config_file: str = "appconfig.json"):
        """
        初始化日志管理器。如果已经初始化过，则跳过。
        :param config_dir: 配置文件所在的目录路径，默认为None（当前目录）
        :param config_file: 配置文件名称，默认为"appconfig.json"
        """
        if self._initialized:
            return
        self._initialized = True
        self.config_dir = config_dir
        self.config_file = config_file
        self.config = self._load_config()
        logger_name = self.config.get('log_name', 'AppLogger')
        self.logger = logging.getLogger(logger_name)
        self._setup_logger()

    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        :return: 包含配置信息的字典
        """
        try:
            config_path = self.config_file
            if self.config_dir:
                config_path = os.path.join(self.config_dir, self.config_file)

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('log', {})
        except Exception as e:
            print(f"无法加载配置文件：{e}")
            return {}

    def _get_log_level(self) -> int:
        """
        根据配置文件获取日志级别
        :return: 日志级别的整数值
        """
        level_str = self.config.get('log_level', 'DEBUG').upper()
        return getattr(logging, level_str, logging.DEBUG)

    def _setup_logger(self):
        """
        根据配置设置日志记录器
        """
        log_level = self._get_log_level()
        self.logger.setLevel(log_level)

        # 避免重复添加处理器
        if self.logger.handlers:
            # 如果已经有处理器，先清空
            self.logger.handlers.clear()

        # 设置控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        formatter = self._create_formatter()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 如果启用了文件日志，设置文件处理器
        if self.config.get('log_file_enabled', False):
            file_handler = self._setup_file_handler()
            if file_handler:
                file_handler.setLevel(log_level)
                self.logger.addHandler(file_handler)

    def _create_formatter(self):
        """
        创建自定义的日志格式化器
        """

        class CustomFormatter(logging.Formatter):
            def __init__(self, time_format):
                super().__init__()
                self.time_format = time_format

            def format(self, record):
                if self.time_format == 'unix':
                    time_str = str(int(record.created * 1000))  # Unix毫秒时间戳
                else:
                    # time_str = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    # 使用 UTC+8 时区
                    utc_dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
                    local_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
                    time_str = local_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                return f"[{record.levelname}] {time_str} {record.getMessage()}"

        # 读取时间格式
        log_time_format = self.config.get('log_time_format', 'regular')
        return CustomFormatter(log_time_format)

    def _setup_file_handler(self) -> Optional[RotatingFileHandler]:
        """
        设置文件日志处理器，支持日志滚动
        :return: 文件处理器对象，如果设置失败则返回None
        """
        try:
            log_folder = self.config.get('log_folder_name', 'log')
            if not os.path.isabs(log_folder) and self.config_dir:
                log_folder = os.path.join(self.config_dir, log_folder)

            if not os.path.exists(log_folder):
                os.makedirs(log_folder)

            # 使用配置的文件名格式或默认格式
            file_name_format = self.config.get('log_file_name_format', '%Y%m%d')
            date_str = datetime.now().strftime(file_name_format)
            log_file = f"{date_str}{self.config.get('log_file_ext', '.log')}"
            file_path = os.path.join(log_folder, log_file)

            # 获取滚动日志的配置
            max_bytes = self.config.get('log_max_size', 10 * 1024 * 1024)  # 默认10MB
            backup_count = self.config.get('log_backup_count', 3)  # 默认保留最近3个备份

            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )

            formatter = self._create_formatter()
            file_handler.setFormatter(formatter)

            return file_handler
        except Exception as e:
            print(f"设置文件日志处理器时出错：{e}")
            return None

    def debug(self, message: str):
        """记录调试级别的日志"""
        self.logger.debug(message)

    def info(self, message: str):
        """记录信息级别的日志"""
        self.logger.info(message)

    def warning(self, message: str):
        """记录警告级别的日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录错误级别的日志"""
        self.logger.error(message)

    def critical(self, message: str):
        """记录严重错误级别的日志"""
        self.logger.critical(message)

    def get_celery_logger(self) -> logging.Logger:
        """
        获取 Celery 专用的 logger 实例
        使用与系统日志相同的配置，但日志文件名前缀为 celery_
        :return: Celery logger 实例
        """
        if self._celery_logger is not None:
            return self._celery_logger

        celery_logger = logging.getLogger('CeleryLogger')
        # 避免重复添加处理器
        if celery_logger.handlers:
            celery_logger.handlers.clear()

        log_level = self._get_log_level()
        celery_logger.setLevel(log_level)

        # 设置控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = self._create_formatter()
        console_handler.setFormatter(formatter)
        celery_logger.addHandler(console_handler)

        # 如果启用了文件日志，设置文件处理器
        if self.config.get('log_file_enabled', False):
            file_handler = self._setup_celery_file_handler()
            if file_handler:
                file_handler.setLevel(log_level)
                celery_logger.addHandler(file_handler)

        self._celery_logger = celery_logger
        return celery_logger

    def _setup_celery_file_handler(self) -> Optional[RotatingFileHandler]:
        """
        设置 Celery 文件日志处理器
        :return: 文件处理器对象，如果设置失败则返回None
        """
        try:
            log_folder = self.config.get('log_folder_name', 'log')
            if not os.path.isabs(log_folder) and self.config_dir:
                log_folder = os.path.join(self.config_dir, log_folder)

            if not os.path.exists(log_folder):
                os.makedirs(log_folder)

            # Celery 日志文件名: celery_YYYYMMDD.log
            file_name_format = self.config.get('log_file_name_format', '%Y%m%d')
            date_str = datetime.now().strftime(file_name_format)
            log_file = f"celery_{date_str}{self.config.get('log_file_ext', '.log')}"
            file_path = os.path.join(log_folder, log_file)

            # 获取滚动日志的配置
            max_bytes = self.config.get('log_max_size', 10 * 1024 * 1024)
            backup_count = self.config.get('log_backup_count', 3)

            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )

            formatter = self._create_formatter()
            file_handler.setFormatter(formatter)

            return file_handler
        except Exception as e:
            print(f"设置 Celery 文件日志处理器时出错：{e}")
            return None

    def setup_celery_root_logger(self):
        """
        配置 Python 根 logger，使所有模块的日志都写入 Celery 日志文件
        用于 Celery worker 启动时调用
        """
        log_level = self._get_log_level()
        formatter = self._create_formatter()

        # 创建处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        file_handler = None
        if self.config.get('log_file_enabled', False):
            file_handler = self._setup_celery_file_handler()
            if file_handler:
                file_handler.setLevel(log_level)

        # 配置根 logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers = []
        root_logger.addHandler(console_handler)
        if file_handler:
            root_logger.addHandler(file_handler)

        # 需要配置的 logger 列表（包括 celery 和应用模块）
        loggers_to_configure = [
            'celery',
            'celery.task',
            'celery.worker',
            'celery.app.trace',
            'celery_app',
            'celery_app.tasks',
            'celery_app.celery_config',
            'agents',
            'agents.slides_agent',
            'services',
            'services.paper2slides_service',
            'services.minio_service',
            'services.task_service',
            'repositories',
            'common',
            'paper2slides',
        ]

        # 确保所有子 logger 都传播到根 logger
        for logger_name in loggers_to_configure:
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)
            logger.handlers = []  # 清除已有处理器
            logger.propagate = True  # 确保传播到根 logger


# 使用示例
if __name__ == "__main__":
    # 创建日志管理器实例，使用默认配置
    # log_manager = LogManager()

    # 创建日志管理器实例，指定配置文件夹和文件名
    log_manager = LogManager(config_dir="../", config_file="appconfig.json")

    # 记录不同级别的日志
    log_manager.debug("这是一条调试日志")
    log_manager.info("这是一条信息日志")
    log_manager.warning("这是一条警告日志")
    log_manager.error("这是一条错误日志")
    log_manager.critical("这是一条严重错误日志")

    # 验证单例模式
    another_log_manager = LogManager()
    print(f"两个实例是否相同: {log_manager is another_log_manager}")


# ==================== 便捷函数 ====================

def get_celery_logger():
    """
    获取 Celery 专用的 logger（便捷函数）
    用于 Celery 任务和相关模块中的日志记录

    :return: Celery logger 实例
    """
    from pathlib import Path
    config_dir = str(Path(__file__).parent.parent)
    log_manager = LogManager(config_dir=config_dir, config_file="appconfig.json")
    return log_manager.get_celery_logger()
