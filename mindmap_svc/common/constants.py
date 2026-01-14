"""常量定义模块"""
import os

# ============ 应用配置 ============

# 应用配置文件名称
APPLICATION_CONFIG_FILE_NAME = "appconfig.json"

# ============ 服务信息 ============

# 服务名称
SERVICE_NAME = "paper-insight-svc"

# 服务标题
SERVICE_TITLE = "学术论文洞悉生成服务"

# 服务描述
SERVICE_DESCRIPTION = "学术论文洞悉生成服务"

# 服务版本
SERVICE_VERSION = "0.9.0"

# ============ API配置 ============

# API版本
API_VERSION = "v1"

# API前缀
API_PREFIX = f"/api/{API_VERSION}"

# ============ 任务配置 ============

# 任务标题模板
TASK_TITLE_POSTER = "报告聚合"
TASK_TITLE_SLIDES = "演示文稿"
TASK_TITLE_MINDMAP = "脑图智绘"

# 任务队列限制
MAX_RUNNING_TASKS = 2
MAX_WAITING_TASKS = 5

# 默认超时时间（秒）
DEFAULT_TASK_TIMEOUT = 600

# ============ 文件配置 ============

# 文件存储路径
UPLOAD_DIR = "sources/uploads"
OUTPUT_DIR = "outputs"
