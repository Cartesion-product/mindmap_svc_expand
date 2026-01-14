"""响应模型定义模块"""
from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, Field

from common.enums import AgentTypeEnum, TaskStatusEnum
from models.base import BaseResponse, PageInfo


class TaskInfo(BaseModel):
    """任务信息"""
    task_id: str = Field(..., description="任务ID")
    title: str = Field(..., description="任务标题")
    agent_type: AgentTypeEnum = Field(..., description="任务类型")
    status: TaskStatusEnum = Field(..., description="任务状态")
    created_time: datetime = Field(..., description="创建时间")
    result_url: Optional[str] = Field(None, description="结果URL")
    image_urls: Optional[List[str]] = Field(None, description="slides图片URL列表")
    error_reason: Optional[str] = Field(None, description="错误原因")

    model_config = {"from_attributes": True}


class TaskCreateResponse(BaseResponse):
    """任务创建响应"""
    data: Optional[dict] = Field(None, description="任务信息")

    @classmethod
    def success(cls, task_id: str, title: str, status: TaskStatusEnum) -> "TaskCreateResponse":
        """创建成功响应"""
        return cls(
            code=200,
            message="任务已创建",
            data={
                "task_id": task_id,
                "title": title,
                "status": status.value
            }
        )


class TaskDetailResponse(BaseResponse):
    """任务详情响应"""
    data: Optional[TaskInfo] = Field(None, description="任务详情")


class TaskListData(BaseModel):
    """任务列表数据"""
    items: List[TaskInfo] = Field(default_factory=list, description="任务列表")
    page_info: PageInfo = Field(..., description="分页信息")


class TaskListResponse(BaseResponse):
    """任务列表响应"""
    data: Optional[TaskListData] = Field(None, description="列表数据")


class TaskCancelResponse(BaseResponse):
    """任务取消响应"""

    @classmethod
    def success(cls, task_id: str) -> "TaskCancelResponse":
        """取消成功响应"""
        return cls(
            code=200,
            message="任务已取消",
            data={"task_id": task_id}
        )


class FileUploadResponse(BaseResponse):
    """文件上传响应"""
    data: Optional[dict] = Field(None, description="上传结果")

    @classmethod
    def success(cls, file_id: str, file_path: str) -> "FileUploadResponse":
        """上传成功响应"""
        return cls(
            code=200,
            message="文件上传成功",
            data={
                "file_id": file_id,
                "file_path": file_path
            }
        )


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field("healthy", description="服务状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="服务版本")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")


class ErrorResponse(BaseResponse):
    """错误响应"""

    @classmethod
    def create(cls, code: int, message: str, detail: Optional[str] = None) -> "ErrorResponse":
        """创建错误响应"""
        data = {"detail": detail} if detail else None
        return cls(code=code, message=message, data=data)
