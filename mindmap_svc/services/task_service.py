"""任务管理服务模块

提供任务创建、删除、查询和队列管理功能。
"""
import logging
import uuid
import json
from typing import Optional, Dict, Any
from functools import lru_cache

from config.settings import get_settings
from common.enums import AgentTypeEnum, TaskStatusEnum, PaperTypeEnum
from common.redis_manager import get_redis_queue_manager, RedisQueueManager
from models.entities.user_paper_result import UserPaperResult
from repositories.user_paper_repo import get_user_paper_repo, UserPaperRepository
from repositories.system_paper_repo import get_system_paper_repo, SystemPaperRepository
from services.minio_service import get_minio_service, MinIOService
from exception.exceptions import (
    TaskQueueFullException,
    TaskNotFoundException,
    InvalidRequestException
)

logger = logging.getLogger(__name__)


class TaskService:
    """任务管理服务类

    提供任务的创建、删除、查询等业务逻辑。
    """

    def __init__(
        self,
        user_repo: UserPaperRepository,
        system_repo: SystemPaperRepository,
        minio_service: MinIOService,
        queue_manager: RedisQueueManager
    ):
        self._user_repo = user_repo
        self._system_repo = system_repo
        self._minio_service = minio_service
        self._queue_manager = queue_manager
        self._settings = get_settings()

    def create_mindmap_task(
            self,
            paper_id: str,
            source: str,
            paper_type: str,
            user_id: str,
            title: str = "脑图智绘",
            prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建脑图任务 (MindMap 专用)"""

        agent_type = AgentTypeEnum.MINDMAP.value

        # 1. 检查队列容量 (共享同一个队列配额)
        action = self._check_queue_capacity()
        result_id = str(uuid.uuid4())
        update_system = False

        # 2. 系统/用户 论文逻辑检查
        if paper_type == PaperTypeEnum.SYSTEM.value:
            # --- 系统论文逻辑 ---
            default_result = self._system_repo.get_default_result(paper_id, agent_type, source)

            if default_result and (default_result.result or default_result.file_path):
                # A. 系统已有结果 -> 直接复用
                # 检查用户是否已存在该任务 (避免重复插入)
                existing_task = self._user_repo.find_user_existing_task(user_id, paper_id, source, agent_type)
                if existing_task:
                    logger.info(f"用户已存在系统论文脑图任务，返回旧ID: {existing_task.result_id}")
                    return {"task_id": existing_task.result_id}

                # 创建新记录并标记成功
                task = UserPaperResult.create(
                    result_id=result_id,
                    agent_type=agent_type,
                    paper_id=paper_id,
                    source=source,
                    paper_type=paper_type,
                    user_id=user_id,
                    title=title,
                    prompt=prompt
                )
                task.mark_success(
                    file_path=default_result.file_path,
                    images=default_result.images,
                    result=default_result.result  # 核心: 复用JSON结构
                )
                self._user_repo.insert(task)
                logger.info(f"脑图任务秒传成功(复用系统结果): {result_id}")
                return {"task_id": result_id}

            else:
                # B. 系统无结果 -> 创建系统占位符，准备生成
                if not default_result:
                    self._system_repo.insert_empty_record(paper_id, source, agent_type)
                update_system = True

        else:
            # --- 用户论文逻辑 (单例限制) ---
            # 检查是否已存在该任务
            existing_task = self._user_repo.find_user_existing_task(user_id, paper_id, source, agent_type)
            if existing_task:
                # 用户论文只能有一份脑图，必须先删除旧的
                raise InvalidRequestException("该论文的脑图已存在，请先删除旧任务后重新生成。")
            update_system = False

        # 3. 创建新任务记录
        task = UserPaperResult.create(
            result_id=result_id,
            agent_type=agent_type,
            paper_id=paper_id,
            source=source,
            paper_type=paper_type,
            user_id=user_id,
            title=title,
            prompt=prompt
        )
        self._user_repo.insert(task)

        # 4. 提交执行
        if action == "run_now":
            task.mark_running()
            self._user_repo.update_task(task)
            self._queue_manager.increment_running()

            # 调用 MindMap 专用的 Celery 提交方法
            self._submit_mindmap_to_celery(
                result_id=result_id,
                paper_id=paper_id,
                source=source,
                paper_type=paper_type,
                user_id=user_id,
                prompt=prompt,
                update_system=update_system
            )
            logger.info(f"脑图任务立即执行: {result_id}")
        else:
            if not self._queue_manager.add_to_waiting_queue(result_id):
                self._user_repo.delete_by_result_id(result_id)  # 回滚
                raise TaskQueueFullException("任务队列已满")
            logger.info(f"脑图任务进入等待队列: {result_id}")

        return {"task_id": result_id}

    def _submit_mindmap_to_celery(self, result_id, paper_id, source, paper_type, user_id, prompt, update_system):
        """提交 Celery 任务 (MindMap)"""
        from celery_app.tasks import generate_mindmap_task

        generate_mindmap_task.apply_async(
            kwargs={
                "result_id": result_id,
                "paper_id": paper_id,
                "source": source,
                "paper_type": paper_type,
                "agent_type": AgentTypeEnum.MINDMAP.value,
                "user_id": user_id,
                "prompt": prompt,
                "update_system": update_system
            },
            task_id=result_id,
            queue="mindmap"
        )

    def _check_queue_capacity(self) -> str:
        """检查队列容量并决定任务执行策略

        Returns:
            "run_now" - 可以立即运行
            "wait_in_queue" - 需要进入等待队列

        Raises:
            TaskQueueFullException: 等待队列已满
        """
        # 使用Redis查询队列状态（性能优化）
        if self._queue_manager.can_run_now():
            return "run_now"

        # 运行中已满，检查等待队列
        waiting_count = len(self._queue_manager.get_waiting_queue())
        max_waiting = self._settings.max_waiting_tasks

        if waiting_count >= max_waiting:
            raise TaskQueueFullException(
                f"任务队列已满，请稍后重试。"
                f"等待中: {waiting_count}/{max_waiting}"
            )

        return "wait_in_queue"

    def schedule_from_waiting_queue(self) -> None:
        """从等待队列调度下一个任务

        如果运行中任务数未满，从等待队列取出任务并提交到Celery。
        """
        if not self._queue_manager.can_run_now():
            return

        task_id = self._queue_manager.schedule_next()
        if not task_id:
            return

        # 获取任务信息
        task = self._user_repo.find_by_result_id(task_id)
        if task is None:
            logger.warning(f"等待队列中的任务不存在: {task_id}")
            return

        if task.status != TaskStatusEnum.WAITING.value:
            logger.warning(f"等待队列中的任务状态异常: {task_id}, status={task.status}")
            return

        # 提交到Celery
        task.mark_running()
        self._user_repo.update_task(task)
        self._queue_manager.increment_running()

        # 使用保存的参数重新提交任务
        self._submit_mindmap_to_celery(
            result_id=task.result_id,
            paper_id=task.paper_id,
            source=task.source,
            paper_type=task.paper_type,
            user_id=task.user_id,
            prompt=task.prompt,
            update_system=False
        )

        logger.info(f"成功调度任务: {task_id}, update_system={task.update_system}")

    def delete_task(self, task_id: str, user_id: str) -> bool:
        """删除任务

        如果任务正在运行，会先尝试取消。

        Args:
            task_id: 任务ID
            user_id: 用户ID

        Returns:
            是否删除成功

        Raises:
            TaskNotFoundException: 任务不存在
        """
        task = self._user_repo.find_by_result_id(task_id)
        if task is None:
            raise TaskNotFoundException(task_id)

        # 验证所有权
        if task.user_id != user_id:
            raise TaskNotFoundException(task_id)

        # 如果任务正在运行，先取消
        if task.status == TaskStatusEnum.RUNNING.value:
            from celery_app.celery_config import celery_app
            celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"已取消运行中的任务: {task_id}")

            # 减少运行中计数并触发调度
            self._queue_manager.decrement_running()
            self.schedule_from_waiting_queue()

        # 从等待队列中移除（如果在等待队列中）
        if task.status == TaskStatusEnum.WAITING.value:
            self._queue_manager.schedule_next()

        # 删除数据库记录（不删除 MinIO 中的文件，不影响系统论文默认结果）
        self._user_repo.delete_task(task_id)
        logger.info(f"任务已删除: {task_id}")

        return True

    def get_task_detail(self, task_id: str, user_id: str) -> Dict[str, Any]:
        """获取任务详情

        Args:
            task_id: 任务ID
            user_id: 用户ID

        Returns:
            任务详情字典

        Raises:
            TaskNotFoundException: 任务不存在
        """
        task = self._user_repo.find_by_result_id(task_id)
        if task is None or task.user_id != user_id:
            raise TaskNotFoundException(task_id)

        return task.to_dict()

    def get_task_download(self, task_id: str, user_id: str) -> Dict[str, Any]:
        """获取任务下载链接

        Args:
            task_id: 任务ID
            user_id: 用户ID

        Returns:
            下载信息 {"file_path": "", "expires_in": 3600}

        Raises:
            TaskNotFoundException: 任务不存在
        """
        task = self._user_repo.find_by_result_id(task_id)
        if task is None or task.user_id != user_id:
            raise TaskNotFoundException(task_id)

        if task.status != TaskStatusEnum.SUCCESS.value or not task.file_path:
            raise InvalidRequestException("任务未完成或无结果文件")

        # 解析桶名和对象名
        file_path = task.file_path
        if "/" in file_path:
            parts = file_path.split("/", 1)
            bucket_name = parts[0]
            object_name = parts[1]
        else:
            raise InvalidRequestException("无效的文件路径格式")

        # 生成预签名 URL
        expires = 3600  # 1小时
        url = self._minio_service.get_file_url(bucket_name, object_name, expires)

        return {
            "file_path": url,
            "expires_in": expires
        }

    def download_mindmap_html(
        self,
        task_id: str,
        user_id: str,
        expires: int = 3600
    ) -> Dict[str, Any]:
        """下载MindMap HTML文件

        Args:
            task_id: 任务ID
            user_id: 用户ID
            expires: 过期时间（秒），默认1小时

        Returns:
            下载信息 {"download_url": "", "expires_in": 3600}

        Raises:
            TaskNotFoundException: 任务不存在
            InvalidRequestException: 任务未完成或无结果数据
        """
        import tempfile
        from pathlib import Path
        from utils.mindmap_utils import MindMapNode, create_html_mindmap_from_object

        task = self._user_repo.find_by_result_id(task_id)
        if task is None or task.user_id != user_id:
            raise TaskNotFoundException(task_id)

        if task.status != TaskStatusEnum.SUCCESS.value:
            raise InvalidRequestException("任务未完成")

        if not task.result:
            raise InvalidRequestException("无结果数据，无法生成HTML")

        try:
            data = json.loads(task.result)
        except json.JSONDecodeError:
            raise InvalidRequestException("结果数据格式错误")

        if not data or "id" not in data or "topic" not in data:
            raise InvalidRequestException("结果数据格式错误")

        root_node = MindMapNode.from_dict(data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_html_path = f.name

        create_html_mindmap_from_object(root_node, temp_html_path)

        upload_result = self._minio_service.upload_mindmap_html(
            agent_type=AgentTypeEnum.MINDMAP.value,
            paper_type=task.paper_type,
            paper_id=task.paper_id,
            source=task.source,
            user_id=task.user_id,
            result_id=task.result_id,
            html_file_path=temp_html_path
        )

        Path(temp_html_path).unlink()

        bucket_name, object_name = upload_result["file_path"].split("/", 1)
        download_url = self._minio_service.get_file_url(bucket_name, object_name, expires)

        return {
            "download_url": download_url,
            "expires_in": expires
        }

    def list_tasks(
        self,
        user_id: str,
        paper_id: str,
        source: str,
        page: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """分页查询任务列表

        Args:
            user_id: 用户ID
            paper_id: 论文ID
            source: 论文来源
            page: 页码（0表示查询所有）
            page_size: 每页大小

        Returns:
            分页结果 {"items": [], "total": 0, "page": 1, "page_size": 10}
        """
        if page == 0:
            # 查询所有
            items, total = self._user_repo.find_by_user_and_paper(
                user_id=user_id,
                paper_id=paper_id,
                source=source,
                skip=0,
                limit=1000  # 设置一个较大的限制
            )
        else:
            skip = (page - 1) * page_size
            items, total = self._user_repo.find_by_user_and_paper(
                user_id=user_id,
                paper_id=paper_id,
                source=source,
                skip=skip,
                limit=page_size
            )

        return {
            "items": [
                {
                    "task_id": item.result_id,
                    "title": item.title,
                    "agent_type": item.agent_type,
                    "status": item.status,
                    "created_time": item.created_time.isoformat() if item.created_time else None,
                    "paper_id": item.paper_id,
                    "source": item.source
                }
                for item in items
            ],
            "total": total,
            "page": page,
            "page_size": page_size
        }

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态

        Returns:
            队列状态 {"running": 0, "waiting": 0, "max_running": 2, "max_waiting": 5}
        """
        return self._queue_manager.get_queue_status()


@lru_cache()
def get_task_service() -> TaskService:
    """获取任务服务单例

    Returns:
        TaskService: 服务实例
    """
    return TaskService(
        user_repo=get_user_paper_repo(),
        system_repo=get_system_paper_repo(),
        minio_service=get_minio_service(),
        queue_manager=get_redis_queue_manager()
    )
