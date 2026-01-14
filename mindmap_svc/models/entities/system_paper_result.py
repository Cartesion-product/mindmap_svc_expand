"""系统论文结果实体模型

存储系统论文的默认生成结果。
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from common.enums import AgentTypeEnum


class SystemPaperResult(BaseModel):
    """系统论文结果实体

    存储系统论文的默认生成结果，每篇论文每种类型只保存一条记录。

    Attributes:
        paper_id: 论文ID
        source: 论文数据源
        agent_type: 任务类型 (mindmap=脑图)
        file_path: 结果文件路径
        # images: 图像地址列表 (slides专用)
        result_id: 关联的任务ID
        created_time: 创建时间
    """

    paper_id: str = Field(..., description="论文ID")
    source: str = Field(..., description="论文数据源")
    agent_type: str = Field(..., description="任务类型 (mindmap)")
    file_path: Optional[str] = Field(None, description="结果文件路径")
    images: Optional[List[str]] = Field(None, description="图像地址列表")
    result_id: Optional[str] = Field(None, description="任务ID")
    result: Optional[str] = Field(None, description="结构化结果数据JSON字符串(如脑图JSON)")
    created_time: Optional[datetime] = Field(None, description="创建时间")

    model_config = {
        "populate_by_name": True,
        "json_encoders": {datetime: lambda v: v.isoformat()},
        "extra": "ignore"
    }

    @classmethod
    def create(
        cls,
        paper_id: str,
        source: str,
        agent_type: str
    ) -> "SystemPaperResult":
        """创建新实例

        Args:
            paper_id: 论文ID
            source: 论文数据源
            agent_type: 任务类型

        Returns:
            SystemPaperResult: 新创建的实例
        """
        return cls(
            paper_id=paper_id,
            source=source,
            agent_type=agent_type
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于数据库存储"""
        return self.model_dump(by_alias=False, exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemPaperResult":
        """从字典创建实例"""
        if "_id" in data:
            data.pop("_id")

        # 处理result字段的兼容性：如果是从数据库加载的字典格式，转换为JSON字符串
        if "result" in data and isinstance(data["result"], dict):
            import json
            data["result"] = json.dumps(data["result"], ensure_ascii=False)

        return cls(**data)

    def update_result(self, file_path: str, images: Optional[List[str]] = None, result_id: str = None) -> None:
        """更新结果

        Args:
            file_path: 文件路径
            images: 图像地址列表
            result_id: 任务ID
        """
        self.file_path = file_path
        if images:
            self.images = images
        if result_id:
            self.result_id = result_id
