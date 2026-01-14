"""自定义异常模块"""
from typing import Optional


class CustomException(Exception):
    """自定义异常基类"""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ServerException(CustomException):
    """服务器内部错误"""

    def __init__(self, message: str = "服务器内部错误"):
        super().__init__(message, 500)


class InvalidRequestException(CustomException):
    """无效请求异常"""

    def __init__(self, message: str = "无效的请求参数"):
        super().__init__(message, 400)


class TotalCallsLimitException(CustomException):
    """调用次数限制异常"""

    def __init__(self, message: str = "调用次数已达上限"):
        super().__init__(message, 429)


class RateLimitException(CustomException):
    """速率限制异常"""

    def __init__(self, message: str = "请求频率过高"):
        super().__init__(message, 429)


class NoAuthException(CustomException):
    """未授权异常"""

    def __init__(self, message: str = "未授权访问"):
        super().__init__(message, 401)


class TaskNotFoundException(CustomException):
    """任务不存在异常"""

    def __init__(self, task_id: str):
        super().__init__(f"任务不存在: {task_id}", 404)


class TaskQueueFullException(CustomException):
    """任务队列已满异常"""

    def __init__(self, message: str = "任务队列已满，请稍后重试"):
        super().__init__(message, 503)


class PaperNotFoundException(CustomException):
    """论文不存在异常"""

    def __init__(self, paper_id: str):
        super().__init__(f"论文不存在: {paper_id}", 404)


class SessionConflictException(CustomException):
    """会话冲突异常"""

    def __init__(self, running_session_id: str):
        super().__init__(
            f"另一个会话正在运行中，请等待完成。运行中的会话: {running_session_id[:8]}",
            409
        )


class FileNotFoundException(CustomException):
    """文件不存在异常"""

    def __init__(self, file_path: str):
        super().__init__(f"文件不存在: {file_path}", 404)


class UnsupportedFileTypeException(CustomException):
    """不支持的文件类型异常"""

    def __init__(self, file_type: str):
        super().__init__(f"不支持的文件类型: {file_type}", 400)
