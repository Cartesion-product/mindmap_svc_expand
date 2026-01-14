"""用户论文任务结果仓库

管理用户的任务执行结果。
"""
import logging
from typing import Optional, List, Tuple, Dict
from functools import lru_cache
from datetime import datetime

from db.mongo import get_mongo_client
from models.entities.user_paper_result import UserPaperResult
from common.enums import TaskStatusEnum
from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class UserPaperRepository(BaseRepository[UserPaperResult]):
    """用户论文任务结果仓库

    管理 user_paper_agent_result 集合的数据访问。
    """

    def __init__(self):
        client = get_mongo_client()
        super().__init__(
            collection=client.user_paper_collection,
            entity_class=UserPaperResult
        )

    def find_by_result_id(self, result_id: str) -> Optional[UserPaperResult]:
        """根据任务ID查找

        Args:
            result_id: 任务ID

        Returns:
            任务结果或 None
        """
        return self.find_by_id(result_id)

    def find_user_existing_task(
        self,
        user_id: str,
        paper_id: str,
        source: str,
        agent_type: str
    ) -> Optional[UserPaperResult]:
        """查找用户是否已有同论文同类型的任务

        Args:
            user_id: 用户ID
            paper_id: 论文ID
            source: 论文来源
            agent_type: 任务类型

        Returns:
            现有任务或 None
        """
        return self.find_one({
            "user_id": user_id,
            "paper_id": paper_id,
            "source": source,
            "agent_type": agent_type
        })

    def has_user_task(
        self,
        user_id: str,
        paper_id: str,
        source: str,
        agent_type: str
    ) -> bool:
        """检查用户是否创建过同论文同类型的任务

        Args:
            user_id: 用户ID
            paper_id: 论文ID
            source: 论文来源
            agent_type: 任务类型

        Returns:
            是否存在任务
        """
        count = self._collection.count_documents({
            "user_id": user_id,
            "paper_id": paper_id,
            "source": source,
            "agent_type": agent_type
        })
        return count > 0

    def find_by_user_and_paper(
        self,
        user_id: str,
        paper_id: str,
        source: str,
        skip: int = 0,
        limit: int = 10
    ) -> Tuple[List[UserPaperResult], int]:
        """根据用户和论文分页查询

        Args:
            user_id: 用户ID
            paper_id: 论文ID
            source: 论文来源
            skip: 跳过数量
            limit: 限制数量

        Returns:
            (任务列表, 总数量)
        """
        filter_dict = {
            "user_id": user_id,
            "paper_id": paper_id,
            "source": source
        }
        total = self.count(filter_dict)
        items = self.find_many(
            filter_dict,
            skip=skip,
            limit=limit,
            sort=[("created_time", -1)]
        )
        return items, total

    def find_by_user_paginated(
        self,
        user_id: str,
        paper_id: Optional[str] = None,
        source: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 10
    ) -> Tuple[List[UserPaperResult], int]:
        """分页查询用户任务

        Args:
            user_id: 用户ID
            paper_id: 论文ID（可选）
            source: 论文来源（可选）
            status: 任务状态（可选）
            skip: 跳过数量
            limit: 限制数量

        Returns:
            (任务列表, 总数量)
        """
        filter_dict = {"user_id": user_id}
        if paper_id:
            filter_dict["paper_id"] = paper_id
        if source:
            filter_dict["source"] = source
        if status is not None:
            filter_dict["status"] = status

        total = self.count(filter_dict)
        items = self.find_many(
            filter_dict,
            skip=skip,
            limit=limit,
            sort=[("created_time", -1)]
        )
        return items, total

    def count_by_status(self, status: str) -> int:
        """统计指定状态的任务数量

        Args:
            status: 任务状态

        Returns:
            任务数量
        """
        return self.count({"status": status})

    def count_running(self) -> int:
        """统计运行中的任务数量

        Returns:
            运行中任务数量
        """
        return self.count_by_status(TaskStatusEnum.RUNNING.value)

    def count_waiting(self) -> int:
        """统计等待中的任务数量

        Returns:
            等待中任务数量
        """
        return self.count_by_status(TaskStatusEnum.WAITING.value)

    def update_status(
        self,
        result_id: str,
        status: str,
        **kwargs
    ) -> int:
        """更新任务状态

        Args:
            result_id: 任务ID
            status: 新状态
            **kwargs: 其他更新字段

        Returns:
            修改数量
        """
        update_dict = {"status": status, **kwargs}
        return self.update_by_id(result_id, update_dict)

    def update_task(self, task: UserPaperResult) -> bool:
        """更新任务状态和结果

        更新字段包括: status, file_path, images, result, error_reason, end_time
        """
        update_data = {
            "status": task.status,
            "file_path": task.file_path,
            "images": task.images,
            "error_reason": task.error_reason,
            "end_time": task.end_time,
            "result": task.result,
            "prompt": task.prompt
        }

        # 执行更新
        result = self.collection.update_one(
            {"result_id": task.result_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

    def update_running_tasks(self, paper_id: str, source: str, agent_type: str, file_path: str,
                             images: List[str] = None, result: str = None):
        """更新同一论文同一类型的所有 running 状态任务

        Args:
            paper_id: 论文ID
            source: 论文来源
            agent_type: 任务类型
            file_path: 文件路径
            images: 图像地址列表

        Returns:
            修改数量
        """
        result = self.collection.update_many(
            {
                "paper_id": paper_id,
                "source": source,
                "agent_type": agent_type,
                "status": TaskStatusEnum.RUNNING.value
            },
            {
                "$set": {
                    "status": TaskStatusEnum.SUCCESS.value,
                    "file_path": file_path,
                    "images": images,
                    "result": result,  # 新增
                    "end_time": datetime.now()
                }
            }
        )
        return result.modified_count

    def mark_running(self, result_id: str) -> int:
        """标记任务为运行中

        Args:
            result_id: 任务ID

        Returns:
            修改数量
        """
        return self.update_status(
            result_id,
            TaskStatusEnum.RUNNING.value,
            start_time=datetime.now()
        )

    def mark_success(
        self,
        result_id: str,
        file_path: str,
        images: Optional[List[str]] = None
    ) -> int:
        """标记任务为成功

        Args:
            result_id: 任务ID
            file_path: 结果文件路径
            images: 图像地址列表

        Returns:
            修改数量
        """
        update_dict = {
            "file_path": file_path,
            "end_time": datetime.now()
        }
        if images:
            update_dict["images"] = images
        return self.update_status(result_id, TaskStatusEnum.SUCCESS.value, **update_dict)

    def mark_failed(self, result_id: str, error_reason: str) -> int:
        """标记任务为失败

        Args:
            result_id: 任务ID
            error_reason: 失败原因

        Returns:
            修改数量
        """
        return self.update_status(
            result_id,
            TaskStatusEnum.FAILED.value,
            error_reason=error_reason,
            end_time=datetime.now()
        )

    def get_next_waiting_task(self) -> Optional[UserPaperResult]:
        """获取下一个等待中的任务

        Returns:
            等待中的任务或 None
        """
        return self.find_one({"status": TaskStatusEnum.WAITING.value})

    def delete_task(self, result_id: str) -> int:
        """删除任务

        Args:
            result_id: 任务ID

        Returns:
            删除数量
        """
        return self.delete_by_id(result_id)

    def delete_by_result_id(self, result_id):
        """根据结果ID删除任务

        Args:
            result_id: 任务结果ID

        Returns:
            删除的数量
        """
        result = self.collection.delete_one({"result_id": result_id})
        return result.deleted_count


@lru_cache()
def get_user_paper_repo() -> UserPaperRepository:
    """获取用户论文仓库单例

    Returns:
        UserPaperRepository: 仓库实例
    """
    return UserPaperRepository()
