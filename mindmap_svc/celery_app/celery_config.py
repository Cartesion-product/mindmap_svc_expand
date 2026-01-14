"""Celery 配置模块

配置 Celery 应用和任务队列。
"""
import os
from pathlib import Path
from celery import Celery
from celery.signals import setup_logging

# 创建 Celery 应用
celery_app = Celery("slide_svc")

# 配置
celery_app.conf.update(
    # Broker 配置 (Redis)
    broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),

    # 任务配置
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,

    # 任务结果配置
    result_expires=3600,  # 结果保存 1 小时
    task_track_started=True,    # 任务开始时返回结果

    # 并发配置
    worker_concurrency=2,  # 最大并发任务数
    worker_prefetch_multiplier=1,  # 预取任务数

    # Worker 池配置 - 使用 threads 池避免 Python 3.13 与 billiard 的兼容性问题
    worker_pool="threads",

    # 任务重试配置
    task_acks_late=True,    # 任务失败时，是否等待结果
    task_reject_on_worker_lost=True,    # 任务失败时，是否拒绝任务

    # 任务路由
    task_routes={
        "celery_app.tasks.generate_mindmap_task": {"queue": "mindmap"},
    },

    # 任务时间限制
    task_soft_time_limit=1800,  # 30 分钟软限制
    task_time_limit=2000,  # 33 分钟硬限制

    # 日志配置 - 禁用 Celery 对根 logger 的劫持，使用自定义配置
    worker_hijack_root_logger=False,
)

# 自动发现任务
celery_app.autodiscover_tasks(["celery_app"])


@setup_logging.connect
def setup_celery_logging(**kwargs):
    """
    配置 Celery 日志，复用 LogManager 的配置
    日志文件存放在 logs 文件夹下，格式为 celery_YYYYMMDD.log
    """
    from utilities.log_manager import LogManager

    # 获取项目根目录
    base_dir = Path(__file__).parent.parent

    # 初始化 LogManager 并配置 Celery 根 logger
    log_manager = LogManager(config_dir=str(base_dir), config_file="appconfig.json")
    log_manager.setup_celery_root_logger()
