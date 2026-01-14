"""演示文稿Agent服务主入口

基于LangGraph的智能演示文稿生成服务。
使用 FastAPI + Celery 构建异步任务处理架构。
"""
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

from models import constants
from utilities.log_manager import LogManager

# 初始化日志组件
logger = LogManager(config_file=constants.APPLICATION_CONFIG_FILE_NAME)


def load_env_from_config(
    config_dir: Optional[str] = None,
    config_file_name: Optional[str] = constants.APPLICATION_CONFIG_FILE_NAME
) -> None:
    """从配置文件加载环境变量

    Args:
        config_dir: 配置文件目录，默认为当前目录
        config_file_name: 配置文件名称，默认为 appconfig.json

    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: JSON格式无效
        KeyError: 缺少必需的配置键
        ValueError: 配置值类型错误
    """
    if config_dir is None:
        config_dir = os.getcwd()

    config_path = os.path.join(config_dir, config_file_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if 'mode' not in config or 'env' not in config:
        raise KeyError("配置必须包含 'mode' 和 'env' 键")

    if not isinstance(config['mode'], str):
        raise ValueError("'mode' 必须是字符串")

    if not isinstance(config['env'], dict):
        raise ValueError("'env' 必须是字典")

    # 开发模式下加载环境变量
    if config['mode'].lower() == "dev":
        for key, value in config['env'].items():
            os.environ[key] = str(value)
        logger.info(f"已加载开发环境变量: {config_path}")
    else:
        logger.info(f"当前模式: {config.get('mode')}，未设置环境变量")


def start_celery_worker() -> subprocess.Popen:
    """启动Celery worker进程

    Returns:
        Celery worker进程对象
    """
    log_dir = Path(os.getcwd()) / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "celery_worker.log"

    celery_cmd = [
        sys.executable, "-m", "celery",
        "-A", "celery_app",
        "worker",
        "-Q", "mindmap",
        "-l", "info",
        "--concurrency=2",
        "--prefetch-multiplier=1",  # 降低预取数量
        "--pool=threads",
        # "--logfile", str(log_file)
    ]

    logger.info(f"正在启动Celery worker...")
    logger.info(f"日志文件: {log_file}")

    process = subprocess.Popen(
        celery_cmd,
        cwd=os.getcwd(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    logger.info(f"Celery worker已启动，PID: {process.pid}")
    return process


def cleanup_celery_worker(process: Optional[subprocess.Popen]) -> None:
    """清理Celery worker进程

    Args:
        process: Celery worker进程对象
    """
    if process and process.poll() is None:
        logger.info(f"正在停止Celery worker (PID: {process.pid})...")
        process.terminate()

        try:
            process.wait(timeout=10)
            logger.info("Celery worker已停止")
        except subprocess.TimeoutExpired:
            logger.warning("Celery worker未在10秒内停止，强制终止")
            process.kill()


if __name__ == '__main__':
    load_env_from_config()

    celery_process = None

    try:
        celery_process = start_celery_worker()

        import time
        time.sleep(2)

        from config.settings import get_settings
        from api.api_server import ApiServer

        settings = get_settings()
        api_server = ApiServer(logger=logger, settings=settings)
        api_server.start()

    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在停止服务...")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        if celery_process:
            cleanup_celery_worker(celery_process)
