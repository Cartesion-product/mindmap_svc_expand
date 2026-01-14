"""系统论文结果仓库

管理系统论文的默认生成结果。
"""
import logging
from typing import Optional, List, Dict
from functools import lru_cache
from datetime import datetime

from db.mongo import get_mongo_client
from models.entities.system_paper_result import SystemPaperResult
from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class SystemPaperRepository(BaseRepository[SystemPaperResult]):
    """系统论文结果仓库

    管理 system_paper_agent_result 集合的数据访问。
    """

    def __init__(self):
        client = get_mongo_client()
        super().__init__(
            collection=client.system_paper_collection,
            entity_class=SystemPaperResult
        )

    def find_by_paper_and_type(
        self,
        paper_id: str,
        agent_type: str,
        source: str
    ) -> Optional[SystemPaperResult]:
        """根据论文ID、任务类型和来源查找

        Args:
            paper_id: 论文ID
            agent_type: 任务类型
            source: 论文来源

        Returns:
            系统论文结果或 None
        """
        return self.find_one({
            "paper_id": paper_id,
            "agent_type": agent_type,
            "source": source
        })

    def upsert_result(
        self,
        paper_id: str,
        source: str,
        agent_type: str,
        file_path: str,
        images: Optional[List[str]] = None,
        result_id: Optional[str] = None
    ) -> str:
        """更新或插入结果

        如果已存在则更新，否则插入。

        Args:
            paper_id: 论文ID
            source: 论文来源
            agent_type: 任务类型
            file_path: 结果文件路径
            images: 图像地址列表
            result_id: 任务ID

        Returns:
            文档ID
        """
        filter_dict = {
            "paper_id": paper_id,
            "agent_type": agent_type,
            "source": source
        }

        update_dict = {
            "file_path": file_path,
            "created_time": datetime.now()
        }
        if images:
            update_dict["images"] = images
        if result_id:
            update_dict["result_id"] = result_id

        # 使用 upsert 操作
        result = self._collection.update_one(
            filter_dict,
            {"$set": update_dict, "$setOnInsert": filter_dict},
            upsert=True
        )

        if result.upserted_id:
            return str(result.upserted_id)
        return paper_id

    def save_default_result(
        self,
        paper_id: str,
        source: str,
        agent_type: str,
        file_path: str,
        images: Optional[List[str]] = None,
        result_id: Optional[str] = None
    ) -> str:
        """保存默认结果

        如果已存在则更新，否则插入。

        Args:
            paper_id: 论文ID
            source: 论文来源
            agent_type: 任务类型
            file_path: 结果文件路径
            images: 图像地址列表
            result_id: 任务ID

        Returns:
            文档ID
        """
        return self.upsert_result(
            paper_id=paper_id,
            source=source,
            agent_type=agent_type,
            file_path=file_path,
            images=images,
            result_id=result_id
        )

    def insert_empty_record(
        self,
        paper_id: str,
        source: str,
        agent_type: str
    ) -> str:
        """插入空的系统记录

        如果系统无记录，则插入空记录（file_path为None）。

        Args:
            paper_id: 论文ID
            source: 论文来源
            agent_type: 任务类型

        Returns:
            文档ID，如果已存在则返回None
        """
        filter_dict = {
            "paper_id": paper_id,
            "agent_type": agent_type,
            "source": source
        }

        # 检查是否已存在
        existing = self._collection.find_one(filter_dict)
        if existing:
            return None

        # 插入新记录（file_path为空）
        data = {
            "paper_id": paper_id,
            "source": source,
            "agent_type": agent_type,
            "file_path": None,
            "images": None,
            "created_time": datetime.now()
        }

        result = self._collection.insert_one(data)
        return str(result.inserted_id)

    def get_default_result(
        self,
        paper_id: str,
        agent_type: str,
        source: str
    ) -> Optional[SystemPaperResult]:
        """获取默认结果

        Args:
            paper_id: 论文ID
            agent_type: 任务类型
            source: 论文来源

        Returns:
            系统论文结果或 None
        """
        return self.find_by_paper_and_type(paper_id, agent_type, source)

    def update_file_path(
        self,
        paper_id: str,
        source: str,
        agent_type: str,
        file_path: str,
        images: List[str] = None,
        result_id: str = None,
        result: str = None  # 新增参数
    ) -> None:
        """更新系统论文结果"""
        update_data = {
            "file_path": file_path,
            "images": images,
            "result_id": result_id,
            "created_time": datetime.now()
        }

        # 如果有 result 数据，也更新进去
        if result is not None:
            update_data["result"] = result

        self.collection.update_one(
            {"paper_id": paper_id, "source": source, "agent_type": agent_type},
            {"$set": update_data},
            upsert=True
        )

    def has_default_result(
        self,
        paper_id: str,
        agent_type: str,
        source: str
    ) -> bool:
        """检查是否有默认结果

        Args:
            paper_id: 论文ID
            agent_type: 任务类型
            source: 论文来源

        Returns:
            是否存在默认结果
        """
        result = self.find_by_paper_and_type(paper_id, agent_type, source)
        return result is not None and result.file_path is not None

    def delete_by_paper_id(
        self,
        paper_id: str,
        agent_type: str,
        source: str
    ) -> int:
        """根据论文ID、任务类型和来源删除系统记录

        Args:
            paper_id: 论文ID
            agent_type: 任务类型
            source: 论文来源

        Returns:
            删除的文档数量
        """
        filter_dict = {
            "paper_id": paper_id,
            "agent_type": agent_type,
            "source": source
        }
        result = self._collection.delete_many(filter_dict)
        return result.deleted_count

    def find_empty_results(self) -> List[SystemPaperResult]:
        """查找所有无结果（file_path为空）的系统记录

        Returns:
            无结果的系统记录列表
        """
        return self.find_many(
            {"file_path": None},
            limit=1000
        )

    def delete_empty_results(self) -> int:
        """删除所有无结果（file_path为空）的系统记录

        Returns:
            删除的文档数量
        """
        result = self._collection.delete_many({"file_path": None})
        return result.deleted_count



@lru_cache()
def get_system_paper_repo() -> SystemPaperRepository:
    """获取系统论文仓库单例

    Returns:
        SystemPaperRepository: 仓库实例
    """
    return SystemPaperRepository()
