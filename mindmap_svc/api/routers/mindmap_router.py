"""MindMap 任务接口路由

独立于 Slides/Poster 的路由模块
"""
from fastapi import APIRouter, Depends, Path
from typing import Dict, Any

from middleware.auth import token_decoder
from models.requests import CreateMindMapRequest
from models.responses import BaseResponse
from services.task_service import TaskService
from repositories.user_paper_repo import get_user_paper_repo
from repositories.system_paper_repo import get_system_paper_repo
from services.minio_service import get_minio_service
from common.redis_manager import get_redis_queue_manager
from common.enums import AgentTypeEnum

router = APIRouter(prefix="/api/v1/mindmap/tasks", tags=["MindMap Tasks"])

def get_task_service():
    """依赖注入 TaskService"""
    return TaskService(
        user_repo=get_user_paper_repo(),
        system_repo=get_system_paper_repo(),
        minio_service=get_minio_service(),
        queue_manager=get_redis_queue_manager()
    )

@router.post("/create", response_model=BaseResponse)
async def create_mindmap_task(
    request: CreateMindMapRequest,
    token_payload: Dict[str, Any] = Depends(token_decoder),
    service: TaskService = Depends(get_task_service)
):
    """创建脑图生成任务

    POST /api/v1/mindmap/tasks/create
    """
    user_id = token_payload['user_id']

    try:
        # 调用 MindMap 专用创建逻辑
        result = service.create_mindmap_task(
            paper_id=request.paper_id,
            source=request.source,
            paper_type=request.paper_type,
            user_id=user_id,
            title=request.title,
            prompt=request.prompt
        )
        return BaseResponse(code=200, message="任务创建成功", data=result)

    except Exception as e:
        # 异常建议由全局 ExceptionHandler 捕获，这里为了保险也可以直接抛出
        raise e

@router.delete("/{task_id}/delete", response_model=BaseResponse)
async def delete_mindmap_task(
    task_id: str = Path(..., description="任务ID"),
    token_payload: Dict[str, Any] = Depends(token_decoder),
    service: TaskService = Depends(get_task_service)
):
    """删除任务

    DELETE /api/v1/mindmap/tasks/{task_id}/delete
    """
    user_id = token_payload['user_id']

    # 复用通用的删除逻辑（停止Celery任务 + 删除DB记录）
    success = service.delete_task(task_id, user_id)

    if success:
        return BaseResponse(code=200, message="任务删除成功")
    else:
        return BaseResponse(code=404, message="任务不存在或无法删除")

@router.get("/{task_id}/detail", response_model=BaseResponse)
async def get_mindmap_task_detail(
    task_id: str = Path(..., description="任务ID"),
    token_payload: Dict[str, Any] = Depends(token_decoder),
    service: TaskService = Depends(get_task_service)
):
    """查询任务详情

    GET /api/v1/mindmap/tasks/{task_id}/detail
    """
    user_id = token_payload['user_id']

    task_data = service.get_task_detail(task_id, user_id)

    if not task_data:
        return BaseResponse(code=404, message="任务不存在")

    return BaseResponse(code=200, message="查询成功", data=task_data)

@router.get("/{task_id}/download", response_model=BaseResponse)
async def download_mindmap_html(
    task_id: str = Path(..., description="任务ID"),
    token_payload: Dict[str, Any] = Depends(token_decoder),
    service: TaskService = Depends(get_task_service)
):
    """下载MindMap HTML文件

    GET /api/v1/mindmap/tasks/{task_id}/download
    """
    user_id = token_payload['user_id']

    result = service.download_mindmap_html(task_id, user_id)
    return BaseResponse(code=200, message="下载链接生成成功", data=result)