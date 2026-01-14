"""请求模型定义模块"""
from typing import Optional, List
from pydantic import BaseModel, Field

from common.enums import (
    AgentTypeEnum,
    ContentTypeEnum,
    OutputTypeEnum,
    StyleTypeEnum,
    TaskStatusEnum
)


class CreateTaskRequest(BaseModel):
    """通用创建任务请求 (保留原有)"""
    paper_id: str = Field(..., description="论文ID")
    source: str = Field(..., description="数据源 (arxiv/acl/neurips)")
    paper_type: str = Field(default="system", description="论文类型 (system/user)")
    title: Optional[str] = Field(None, description="任务标题")
    style: str = Field(default="academic", description="风格")
    language: str = Field(default="ZH", description="语言")
    density: str = Field(default="medium", description="密度")


class CreateMindMapRequest(BaseModel):
    """创建脑图任务请求"""
    paper_id: str = Field(..., description="论文ID")
    source: str = Field(..., description="数据源 (arxiv/acl/neurips)")
    paper_type: str = Field(default="system", description="论文类型 (system/user)")
    title: str = Field(default="脑图智绘", description="任务标题")
    prompt: Optional[str] = Field(None, description="用户额外提示词")


class TaskCreateRequest(BaseModel):
    """任务创建请求"""
    paper_id: str = Field(..., description="论文ID")
    agent_type: AgentTypeEnum = Field(..., description="任务类型：10-全景信息图，20-演示文稿")
    content_type: ContentTypeEnum = Field(
        default=ContentTypeEnum.PAPER,
        description="内容类型"
    )
    # output_type: OutputTypeEnum = Field(
    #     default=OutputTypeEnum.SLIDES,
    #     description="输出类型"
    # )
    style: StyleTypeEnum = Field(
        default=StyleTypeEnum.DORAEMON,
        description="风格类型"
    )
    custom_style: Optional[str] = Field(
        None,
        description="自定义风格描述（当style为custom时使用）"
    )
    length: Optional[str] = Field(
        "medium",
        description="slides长度: short/medium/long"
    )
    density: Optional[str] = Field(
        "medium",
        description="poster密度: sparse/medium/dense"
    )
    fast_mode: bool = Field(
        True,
        description="快速模式（仅paper类型有效）"
    )
    message: Optional[str] = Field(
        None,
        description="用户附加消息"
    )


class TaskQueryRequest(BaseModel):
    """任务查询请求"""
    user_id: Optional[str] = Field(None, description="用户ID")
    paper_id: Optional[str] = Field(None, description="论文ID")
    status: Optional[TaskStatusEnum] = Field(None, description="任务状态")
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(10, ge=1, le=100, description="每页大小")


class TaskCancelRequest(BaseModel):
    """任务取消请求"""
    task_id: str = Field(..., description="任务ID")
    reason: Optional[str] = Field(None, description="取消原因")


class PaperUploadRequest(BaseModel):
    """论文上传请求"""
    filename: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件类型")
    file_size: int = Field(..., ge=0, description="文件大小（字节）")


class StylePreviewRequest(BaseModel):
    """风格预览请求"""
    style: StyleTypeEnum = Field(..., description="风格类型")
    custom_style: Optional[str] = Field(None, description="自定义风格描述")
    sample_text: Optional[str] = Field(None, description="示例文本")
