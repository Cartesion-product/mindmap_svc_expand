"""用户论文任务结果实体模型

存储用户的任务执行结果。
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from common.enums import AgentTypeEnum, TaskStatusEnum
from common.constants import TASK_TITLE_MINDMAP


class UserPaperResult(BaseModel):
    """用户论文任务结果实体

    存储用户每次任务的执行结果。

    Attributes:
        result_id: 任务ID（主键）
        title: 任务标题
        agent_type: 任务类型 (mindmap=脑图)
        status: 任务状态 (waiting/running/success/failed)
        error_reason: 失败原因
        paper_id: 论文ID
        source: 论文数据源
        paper_type: 论文类型 (system/user)
        style: 风格偏好
        language: 语言 (ZH/EN)
        density: 内容密度 (sparse/medium/dense)
        file_path: 结果文件路径
        images: 图像地址列表 (slides专用)
        start_time: 开始时间
        end_time: 结束时间
        user_id: 用户ID
        created_time: 创建时间
    """

    result_id: str = Field(..., description="任务ID（主键）")
    title: str = Field(..., description="任务标题")
    agent_type: str = Field(..., description="任务类型 (mindmap)")
    status: str = Field(default=TaskStatusEnum.WAITING.value, description="任务状态")
    error_reason: Optional[str] = Field(None, description="失败原因")
    paper_id: str = Field(..., description="论文ID")
    source: str = Field(..., description="论文数据源")
    paper_type: str = Field(..., description="论文类型 (system/user)")
    style: Optional[str] = Field("academic", description="风格偏好")
    language: str = Field("ZH", description="语言 (ZH/EN)")
    density: str = Field("medium", description="内容密度 (sparse/medium/dense)")
    file_path: Optional[str] = Field(None, description="结果文件路径")
    images: Optional[List[str]] = Field(None, description="图像地址列表")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    user_id: str = Field(..., description="用户ID")
    created_time: datetime = Field(default_factory=datetime.now, description="创建时间")
    result: Optional[str] = Field(None, description="结构化结果数据JSON字符串(如脑图JSON)")
    prompt: Optional[str] = Field(None, description="用户额外提示词")

    model_config = {
        "populate_by_name": True,
        "json_encoders": {datetime: lambda v: v.isoformat() if v else None},
        "extra": "ignore"
    }

    @classmethod
    def create(
        cls,
        result_id: str,
        agent_type: str,
        paper_id: str,
        source: str,
        paper_type: str,
        user_id: str,
        title: Optional[str] = None,
        style: str = "academic",
        language: str = "ZH",
        density: str = "medium",
        prompt: Optional[str] = None,  # 新增参数
    ) -> "UserPaperResult":
        """创建新任务实例

        Args:
            result_id: 任务ID
            agent_type: 任务类型
            paper_id: 论文ID
            source: 论文数据源
            paper_type: 论文类型
            user_id: 用户ID
            title: 任务标题（可选）
            style: 风格偏好
            language: 语言 (ZH/EN)
            density: 内容密度 (sparse/medium/dense)

        Returns:
            UserPaperResult: 新创建的任务实例
        """
        if title is None:
            if agent_type == AgentTypeEnum.MINDMAP.value:
                title = TASK_TITLE_MINDMAP
        return cls(
            result_id=result_id,
            title=title,
            agent_type=agent_type,
            status=TaskStatusEnum.WAITING.value,
            paper_id=paper_id,
            source=source,
            paper_type=paper_type,
            user_id=user_id,
            style=style,
            language=language,
            density=density,
            prompt=prompt,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于数据库存储"""
        data = self.model_dump()
        # 使用 result_id 作为 _id
        data["_id"] = data["result_id"]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPaperResult":
        """从字典创建实例"""
        if "_id" in data and "result_id" not in data:
            data["result_id"] = str(data.pop("_id"))
        elif "_id" in data:
            data.pop("_id")

        # 处理result字段的兼容性：如果是从数据库加载的字典格式，转换为JSON字符串
        if "result" in data and isinstance(data["result"], dict):
            import json
            data["result"] = json.dumps(data["result"], ensure_ascii=False)

        return cls(**data)

    def mark_running(self) -> None:
        """标记为运行中"""
        self.status = TaskStatusEnum.RUNNING.value
        self.start_time = datetime.now()

    def mark_success(self, file_path: str, images: Optional[List[str]] = None, result: Optional[str] = None) -> None:
        """标记为成功

        Args:
            file_path: 结果文件路径
            images: 图像地址列表
            result: 结构化结果数据JSON字符串
        """
        self.status = TaskStatusEnum.SUCCESS.value
        self.file_path = file_path
        self.images = images
        self.result = result
        self.end_time = datetime.now()

    def mark_failed(self, error_reason: str) -> None:
        """标记为失败

        Args:
            error_reason: 失败原因
        """
        self.status = TaskStatusEnum.FAILED.value
        self.error_reason = error_reason
        self.end_time = datetime.now()

    @property
    def is_running(self) -> bool:
        """是否运行中"""
        return self.status == TaskStatusEnum.RUNNING.value

    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.status in (TaskStatusEnum.SUCCESS.value, TaskStatusEnum.FAILED.value)