from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from typing import Any, Dict, Union

from exception.exceptions import CustomException, ServerException, InvalidRequestException, TotalCallsLimitException, \
    RateLimitException, NoAuthException


# 异常处理器
async def exception_handler(
        request: Request,
        exc: Union[CustomException, Exception]
) -> JSONResponse:
    """
    统一的异常处理器，处理自定义异常和基础异常

    Args:
        request: FastAPI请求对象
        exc: 异常对象，可以是CustomException或基础Exception

    Returns:
        JSONResponse: 符合OpenAI标准的错误响应
    """
    if isinstance(exc, CustomException):
        # 处理自定义异常
        error_response = {
            "code": exc.status_code,
            "data": None,
            "message": exc.message
        }
        status_code = status.HTTP_200_OK
    else:
        # 处理基础异常
        error_response = {
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "data": None,
            "message": str(exc) or "服务器内部错误"
        }
        status_code = status.HTTP_200_OK

    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    在FastAPI应用中注册异常处理器

    Args:
        app: FastAPI应用实例
    """
    # 注册自定义异常处理器
    custom_exceptions = [
        ServerException,
        InvalidRequestException,
        TotalCallsLimitException,
        RateLimitException,
        NoAuthException
    ]

    for exception in custom_exceptions:
        app.add_exception_handler(exception, exception_handler)

    # 注册基础异常处理器
    app.add_exception_handler(Exception, exception_handler)
