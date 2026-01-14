"""配置管理模块

提供统一的配置管理，从 appconfig.json 加载配置和环境变量。
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache


class Settings:
    """统一配置管理类

    单例模式，负责加载和管理应用配置。
    """

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """加载配置文件"""
        config_path = Path(__file__).parent.parent / "appconfig.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)

    # ============ 基础配置 ============

    @property
    def port(self) -> int:
        """服务端口号"""
        return self._config.get('port', 5003)

    @property
    def mode(self) -> str:
        """运行模式: dev/prod"""
        return self._config.get('mode', 'prod')

    @property
    def is_dev(self) -> bool:
        """是否为开发模式"""
        return self.mode.lower() == 'dev'

    @property
    def log_config(self) -> Dict:
        """日志配置"""
        return self._config.get('log', {})

    # ============ MongoDB 配置 ============

    @property
    def mongo_uri(self) -> str:
        """MongoDB 连接 URI"""
        return os.getenv('KB_FRAMEWORK_DB', '')

    @property
    def mongo_database(self) -> str:
        """MongoDB 数据库名"""
        return os.getenv('SLIDES_MONGO_DB', 'slide_svc')

    @property
    def knowledge_db_name(self) -> str:
        """知识库数据库名 (Source Content)"""
        return os.getenv('KNOWLEDGE_DB_NAME', 'SV_KNOWLEDGE_DB')

    @property
    def system_paper_collection(self) -> str:
        """系统论文结果集合名"""
        return os.getenv('SLIDES_SYSTEM_PAPER_COLLECTION', 'system_paper_agent_result')

    @property
    def user_paper_collection(self) -> str:
        """用户论文结果集合名"""
        return os.getenv('SLIDES_USER_PAPER_COLLECTION', 'user_paper_agent_result')

    @property
    def system_paper_content_collection(self) -> str:
        """系统论文原始内容表名 (Source Content)"""
        return os.getenv('SYSTEM_PAPER_CONTENT_COLLECTION', 'system_paper_original_content')

    # ============ Celery / Redis 配置 ============

    def _get_redis_url_from_kb_env(self, db_index: int = 0) -> Optional[str]:
        """从 KB_REDIS_DB 环境变量解析并构建 Redis URL

        环境变量格式示例: 10.0.62.230:26379,password=SophyVerse-v5
        目标 URL 格式: redis://:password@host:port/db
        """
        kb_redis_conf = os.getenv('KB_REDIS_DB')
        if not kb_redis_conf:
            return None

        try:
            # 1. 按逗号拆分
            parts = kb_redis_conf.split(',')
            host_port = parts[0].strip()

            password = ""
            # 2. 查找 password 部分
            if len(parts) > 1:
                for part in parts[1:]:
                    part = part.strip()
                    if part.startswith("password="):
                        password = part.split("=", 1)[1]
                        break

            # 3. 构建 URL
            # 如果有密码，URL格式为 redis://:password@host:port/db
            # 如果无密码，URL格式为 redis://host:port/db
            auth_str = f":{password}@" if password else ""

            return f"redis://{auth_str}{host_port}/{db_index}"

        except Exception as e:
            # 解析失败时记录错误（或者仅静默失败回退到默认值）
            print(f"Warning: 解析 KB_REDIS_DB 环境变量失败: {e}")
            return None

    @property
    def celery_broker_url(self) -> str:
        """Celery Broker URL (Redis)"""
        # 优先尝试从 KB_REDIS_DB 解析 (默认使用 db 0)
        kb_url = self._get_redis_url_from_kb_env(0)
        if kb_url:
            return kb_url

        return os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')

    @property
    def celery_result_backend(self) -> str:
        """Celery Result Backend (Redis)"""
        # 优先尝试从 KB_REDIS_DB 解析 (默认使用 db 1)
        kb_url = self._get_redis_url_from_kb_env(1)
        if kb_url:
            return kb_url

        return os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

    # ============ MinIO 配置 ============

    @property
    def minio_endpoint(self) -> str:
        """MinIO 服务端点"""
        return os.getenv('KB_MINIO_ENDPOINT', '')

    @property
    def minio_access_key(self) -> str:
        """MinIO 访问密钥"""
        return os.getenv('KB_MINIO_ACCESSKEY', '')

    @property
    def minio_secret_key(self) -> str:
        """MinIO 秘密密钥"""
        return os.getenv('KB_MINIO_SECRETKEY', '')

    # MinIO 多桶配置 - 按类型区分
    @property
    def system_mindmap_bucket(self) -> str:
        """系统 MindMap 桶名"""
        return os.getenv('SYSTEM_MINDMAP_MINIO_BUCKET', 'kb-mindmap-system')

    @property
    def user_mindmap_bucket(self) -> str:
        """用户 MindMap 桶名"""
        return os.getenv('USER_MINDMAP_MINIO_BUCKET', 'kb-mindmap-user')

    def get_bucket_name(self, agent_type: str, paper_type: str) -> str:
        """根据任务类型和论文类型获取对应的桶名

        Args:
            agent_type: 任务类型 (poster/slides/mindmap)
            paper_type: 论文类型 (system/user)

        Returns:
            str: 对应的桶名
        """
        bucket_map = {
            ('mindmap', 'system'): self.system_mindmap_bucket,
            ('mindmap', 'user'): self.user_mindmap_bucket,
        }
        return bucket_map.get((agent_type, paper_type), self.user_mindmap_bucket)

    # ============ 任务队列配置 ============

    @property
    def max_running_tasks(self) -> int:
        """最大并行运行任务数"""
        return int(os.getenv('SLIDES_MAX_RUNNING_TASKS', '2'))

    @property
    def max_waiting_tasks(self) -> int:
        """最大等待队列任务数"""
        return int(os.getenv('SLIDES_MAX_WAITING_TASKS', '5'))

    @property
    def reset_waiting_on_restart(self) -> bool:
        """服务重启时是否将等待队列任务标记为失败

        - True: 将等待任务标记为失败
        - False: 保留等待任务并重新调度（默认）
        """
        return os.getenv('SLIDES_RESET_WAITING_ON_RESTART', 'false').lower() == 'true'

    # ============ LLM 配置 ============

    @property
    def llm_model(self) -> str:
        """LLM 模型名称"""
        return os.getenv('LLM_MODEL', 'gemini-3-flash')

    @property
    def llm_api_key(self) -> str:
        """LLM API 密钥"""
        return os.getenv('RAG_LLM_API_KEY', '')

    @property
    def llm_base_url(self) -> str:
        """LLM API 基础 URL"""
        return os.getenv('RAG_LLM_BASE_URL', '')

    @property
    def llm_max_tokens(self) -> int:
        """LLM 最大 token 数"""
        return int(os.getenv('RAG_LLM_MAX_TOKENS', '16000'))

    # ============ 图像生成配置 ============

    @property
    def image_gen_provider(self) -> str:
        """图像生成服务提供商"""
        return os.getenv('IMAGE_GEN_PROVIDER', 'doubao')

    @property
    def image_gen_api_key(self) -> str:
        """图像生成 API 密钥"""
        return os.getenv('IMAGE_GEN_API_KEY', '')

    @property
    def image_gen_model(self) -> str:
        """图像生成模型"""
        return os.getenv('IMAGE_GEN_MODEL', 'doubao-seedream-4-5-251128')

    # ============ 解析器配置 ============

    @property
    def parser(self) -> str:
        """文档解析器类型"""
        return os.getenv('PARSER', 'mineru')

    @property
    def parser_enabled(self) -> bool:
        """是否启用文档解析"""
        return os.getenv('PARSER_ENABLED', 'False').lower() == 'true'

    # ============ Paper2Slides 服务配置 ============

    @property
    def paper2slides_api_url(self) -> str:
        """Paper2Slides API 地址"""
        return os.getenv('PAPER2SLIDES_API_URL', 'http://localhost:5003/p2s')

    # ============ 认证配置 ============

    @property
    def auth_service_url(self) -> str:
        """认证服务 URL"""
        return os.getenv('KB_API_SERVICE_AUTHENTICATION', '')

    @property
    def global_token(self) -> str:
        """全局访问 Token"""
        return os.getenv('KB_API_SERVICE_GLOBAL_TOKEN', '')


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例

    Returns:
        Settings: 配置管理实例
    """
    return Settings()
