"""基础仓库模块

提供数据仓库的基础类和通用方法。
"""
import logging
from typing import TypeVar, Generic, Optional, List, Dict, Any, Type
from abc import ABC, abstractmethod

from pymongo.collection import Collection
from pymongo.results import InsertOneResult, UpdateResult, DeleteResult

from db.mongo import get_mongo_client

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """基础仓库抽象类

    提供通用的 CRUD 操作接口。
    """

    def __init__(self, collection: Collection, entity_class: Type[T]):
        """初始化仓库

        Args:
            collection: MongoDB 集合
            entity_class: 实体类
        """
        self._collection = collection
        self._entity_class = entity_class

    @property
    def collection(self) -> Collection:
        """获取集合"""
        return self._collection

    def _to_entity(self, data: Optional[Dict[str, Any]]) -> Optional[T]:
        """将字典转换为实体

        Args:
            data: 数据库文档

        Returns:
            实体实例或 None
        """
        if data is None:
            return None
        return self._entity_class.from_dict(data)

    def _to_entities(self, cursor) -> List[T]:
        """将游标转换为实体列表

        Args:
            cursor: MongoDB 游标

        Returns:
            实体列表
        """
        return [self._to_entity(doc) for doc in cursor]

    def find_by_id(self, id_value: str) -> Optional[T]:
        """根据 ID 查找

        Args:
            id_value: ID 值

        Returns:
            实体实例或 None
        """
        doc = self._collection.find_one({"_id": id_value})
        return self._to_entity(doc)

    def find_one(self, filter_dict: Dict[str, Any]) -> Optional[T]:
        """查找单个文档

        Args:
            filter_dict: 过滤条件

        Returns:
            实体实例或 None
        """
        doc = self._collection.find_one(filter_dict)
        return self._to_entity(doc)

    def find_many(
        self,
        filter_dict: Dict[str, Any],
        skip: int = 0,
        limit: int = 0,
        sort: Optional[List[tuple]] = None
    ) -> List[T]:
        """查找多个文档

        Args:
            filter_dict: 过滤条件
            skip: 跳过数量
            limit: 限制数量
            sort: 排序条件

        Returns:
            实体列表
        """
        cursor = self._collection.find(filter_dict)
        if sort:
            cursor = cursor.sort(sort)
        if skip > 0:
            cursor = cursor.skip(skip)
        if limit > 0:
            cursor = cursor.limit(limit)
        return self._to_entities(cursor)

    def count(self, filter_dict: Dict[str, Any]) -> int:
        """统计文档数量

        Args:
            filter_dict: 过滤条件

        Returns:
            文档数量
        """
        return self._collection.count_documents(filter_dict)

    def insert(self, entity: T) -> str:
        """插入文档

        Args:
            entity: 实体实例

        Returns:
            插入的文档 ID
        """
        data = entity.to_dict()
        result: InsertOneResult = self._collection.insert_one(data)
        return str(result.inserted_id)

    def update(self, filter_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> int:
        """更新文档

        Args:
            filter_dict: 过滤条件
            update_dict: 更新内容

        Returns:
            修改的文档数量
        """
        result: UpdateResult = self._collection.update_one(
            filter_dict,
            {"$set": update_dict}
        )
        return result.modified_count

    def update_by_id(self, id_value: str, update_dict: Dict[str, Any]) -> int:
        """根据 ID 更新文档

        Args:
            id_value: ID 值
            update_dict: 更新内容

        Returns:
            修改的文档数量
        """
        return self.update({"_id": id_value}, update_dict)

    def delete(self, filter_dict: Dict[str, Any]) -> int:
        """删除文档

        Args:
            filter_dict: 过滤条件

        Returns:
            删除的文档数量
        """
        result: DeleteResult = self._collection.delete_one(filter_dict)
        return result.deleted_count

    def delete_by_id(self, id_value: str) -> int:
        """根据 ID 删除文档

        Args:
            id_value: ID 值

        Returns:
            删除的文档数量
        """
        return self.delete({"_id": id_value})

    def upsert(self, filter_dict: Dict[str, Any], entity: T) -> str:
        """插入或更新文档

        Args:
            filter_dict: 过滤条件
            entity: 实体实例

        Returns:
            文档 ID
        """
        data = entity.to_dict()
        result = self._collection.update_one(
            filter_dict,
            {"$set": data},
            upsert=True
        )
        return str(result.upserted_id) if result.upserted_id else None
