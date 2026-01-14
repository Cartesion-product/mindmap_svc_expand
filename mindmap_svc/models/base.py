"""基础模型定义模块"""
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class TimestampMixin(BaseModel):
    """时间戳混入类"""
    created_time: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_time: Optional[datetime] = Field(None, description="更新时间")


class UserMixin(BaseModel):
    """用户信息混入类"""
    user_id: str = Field(..., description="用户ID")
    creator: str = Field(..., description="创建者")
    namespace: str = Field(..., description="命名空间")


class BaseResponse(BaseModel):
    """基础响应模型"""
    code: int = Field(200, description="响应状态码")
    message: str = Field("success", description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")

    model_config = {"from_attributes": True}


class PageInfo(BaseModel):
    """分页信息"""
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    total_pages: int = Field(..., description="总页数")

    @classmethod
    def create(cls, total: int, page: int, page_size: int) -> "PageInfo":
        """创建分页信息"""
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
