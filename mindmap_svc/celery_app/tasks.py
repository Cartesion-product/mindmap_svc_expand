"""Celery 任务定义

定义演示文稿生成的异步任务。
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celery_app.celery_config import celery_app
from celery import states
from celery.exceptions import Ignore
from common.redis_manager import get_redis_queue_manager
from utilities.log_manager import get_celery_logger

# 使用 LogManager 的 Celery logger
logger = get_celery_logger()


from agents.mindmap_agent import run_mindmap_agent


@celery_app.task(name="generate_mindmap_task", queue="mindmap")
def generate_mindmap_task(**kwargs):
    """MindMap 生成任务 (单独的方法)"""
    logger.info(f"开始生成任务: {kwargs}")
    asyncio.run(run_mindmap_agent(**kwargs))


def _schedule_next_task(completed_task_id: str) -> None:
    """调度下一个等待中的任务

    Args:
        completed_task_id: 已完成的任务ID（用于日志）
    """
    try:
        from services.task_service import get_task_service
        task_service = get_task_service()
        task_service.schedule_from_waiting_queue()
    except Exception as e:
        logger.error(f"调度下一个任务失败: {e}")


def _cleanup_temp_files(result_id: str) -> None:
    """清理临时文件

    Args:
        result_id: 任务ID
    """
    import shutil

    temp_dir = PROJECT_ROOT / "data" / "temp" / result_id
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"已清理临时文件: {result_id}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {result_id}, 错误: {e}")


@celery_app.task(name="celery_app.tasks.cancel_task")
def cancel_task(result_id: str) -> bool:
    """取消任务

    Args:
        result_id: 任务ID

    Returns:
        是否成功取消
    """
    from repositories.user_paper_repo import get_user_paper_repo

    queue_manager = get_redis_queue_manager()
    user_repo = get_user_paper_repo()
    task = user_repo.find_by_result_id(result_id)

    if task is None:
        logger.warning(f"任务不存在: {result_id}")
        return False

    # 尝试撤销 Celery 任务
    celery_app.control.revoke(result_id, terminate=True, signal="SIGTERM")

    # 更新任务状态为失败
    user_repo.mark_failed(result_id, "任务已被用户取消")

    # 如果任务正在运行，减少计数并触发调度
    from common.enums import TaskStatusEnum
    if task.status == TaskStatusEnum.RUNNING.value:
        queue_manager.decrement_running()
        _schedule_next_task(result_id)

    # 清理临时文件
    _cleanup_temp_files(result_id)

    logger.info(f"任务已取消: {result_id}")
    return True
