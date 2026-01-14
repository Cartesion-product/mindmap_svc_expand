"""枚举定义模块"""
from enum import Enum


class APIMethodEnum(str, Enum):
    """HTTP请求方法枚举"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class AgentTypeEnum(str, Enum):
    """智能体类型枚举"""
    # POSTER = "poster"    # 全景信息图
    # SLIDES = "slides"    # 演示文稿
    MINDMAP = "mindmap"  # 脑图


class TaskStatusEnum(str, Enum):
    """任务状态枚举"""
    WAITING = "waiting"    # 等待中
    RUNNING = "running"    # 运行中
    SUCCESS = "success"    # 成功
    FAILED = "failed"      # 失败


class PaperTypeEnum(str, Enum):
    """论文类型枚举"""
    SYSTEM = "system"      # 系统论文
    USER = "user"          # 个人论文


class ContentTypeEnum(str, Enum):
    """内容类型枚举"""
    PAPER = "paper"
    GENERAL = "general"


class OutputTypeEnum(str, Enum):
    """输出类型枚举"""
    # SLIDES = "slides"
    # POSTER = "poster"


class StyleTypeEnum(str, Enum):
    """风格类型枚举"""
    ACADEMIC = "academic"
    DORAEMON = "doraemon"
    CUSTOM = "custom"


class LanguageEnum(str, Enum):
    """语言枚举"""
    ZH = "ZH"    # 中文
    EN = "EN"    # 英文


class DensityEnum(str, Enum):
    """内容密度枚举"""
    SPARSE = "sparse"    # 稀疏
    MEDIUM = "medium"    # 中等
    DENSE = "dense"      # 密集
