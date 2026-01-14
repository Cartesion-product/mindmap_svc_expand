"""MinIO 文件服务模块

提供文件上传、下载和管理功能，支持多桶配置。
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache
from datetime import timedelta

from minio import Minio
from minio.error import S3Error

from config.settings import get_settings
from common.enums import AgentTypeEnum, PaperTypeEnum

logger = logging.getLogger(__name__)


class MinIOService:
    """MinIO 文件服务类

    提供文件上传、下载和 URL 生成功能，支持多桶配置。
    """

    _instance: Optional['MinIOService'] = None
    _client: Optional[Minio] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self) -> Minio:
        """获取 MinIO 客户端"""
        if self._client is None:
            settings = get_settings()
            # 处理 endpoint，去掉可能存在的协议前缀
            endpoint = settings.minio_endpoint
            if endpoint.startswith('http://'):
                endpoint = endpoint[7:]
            elif endpoint.startswith('https://'):
                endpoint = endpoint[8:]
            logger.info(f"初始化MinIO客户端: endpoint={endpoint}, access_key={settings.minio_access_key[:5]}...")
            self._client = Minio(
                endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=False  # HTTP连接，非HTTPS
            )
            logger.info("MinIO客户端初始化成功")
        return self._client

    def _ensure_bucket(self, bucket_name: str) -> None:
        """确保存储桶存在

        Args:
            bucket_name: 桶名称
        """
        try:
            client = self._get_client()
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"已创建 MinIO bucket: {bucket_name}")
        except S3Error as e:
            error_msg = f"创建 bucket 失败: {e.message if e.message else str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def upload_file(
        self,
        bucket_name: str,
        local_path: str,
        object_name: str,
        content_type: Optional[str] = None
    ) -> str:
        """上传文件到 MinIO

        Args:
            bucket_name: 桶名称
            local_path: 本地文件路径
            object_name: 对象名称（MinIO 中的路径）
            content_type: 文件类型

        Returns:
            上传后的对象路径
        """
        self._ensure_bucket(bucket_name)
        client = self._get_client()

        try:
            client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=local_path,
                content_type=content_type
            )
            logger.info(f"文件已上传: {bucket_name}/{object_name}")
            return object_name
        except S3Error as e:
            error_msg = f"上传文件失败: {e.message if e.message else str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def download_file(
        self,
        bucket_name: str,
        object_name: str,
        local_path: str
    ) -> str:
        """从 MinIO 下载文件

        Args:
            bucket_name: 桶名称
            object_name: 对象名称
            local_path: 本地保存路径

        Returns:
            str: 本地文件路径
        """
        client = self._get_client()

        try:
            # 确保目标目录存在
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=local_path
            )
            logger.info(f"文件已下载: {bucket_name}/{object_name} -> {local_path}")
            return local_path
        except S3Error as e:
            error_msg = f"下载文件失败: {e.message if e.message else str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def get_file_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: int = 7 * 24 * 60 * 60
    ) -> str:
        """获取文件下载 URL

        Args:
            bucket_name: 桶名称
            object_name: 对象名称
            expires: 过期时间（秒），默认7天

        Returns:
            预签名下载 URL
        """
        client = self._get_client()

        try:
            url = client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_name,
                expires=timedelta(seconds=expires)
            )
            return url
        except S3Error as e:
            error_msg = f"获取文件 URL 失败: {e.message if e.message else str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def get_storage_path(self, bucket_name: str, object_name: str) -> str:
        """获取存储路径（不含host，只有桶及资源路径）

        Args:
            bucket_name: 桶名称
            object_name: 对象名称

        Returns:
            str: 存储路径 (bucket_name/object_name)
        """
        return f"{bucket_name}/{object_name}"

    def delete_file(self, bucket_name: str, object_name: str) -> bool:
        """删除文件

        Args:
            bucket_name: 桶名称
            object_name: 对象名称

        Returns:
            是否删除成功
        """
        client = self._get_client()

        try:
            client.remove_object(
                bucket_name=bucket_name,
                object_name=object_name
            )
            logger.info(f"文件已删除: {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            error_msg = f"删除文件失败: {e.message if e.message else str(e)}"
            logger.error(error_msg)
            return False

    def upload_mindmap_html(
        self,
        agent_type: str,
        paper_type: str,
        paper_id: str,
        source: str,
        user_id: str,
        result_id: str = None,
        html_file_path: str = None
    ) -> Dict[str, Any]:
        """上传MindMap HTML文件

        Args:
            agent_type: 任务类型 (mindmap)
            paper_type: 论文类型 (system/user)
            paper_id: 论文ID
            source: 论文来源
            user_id: 用户ID
            result_id: 任务ID（仅用户任务需要）
            html_file_path: 本地HTML文件路径

        Returns:
            上传结果 {"file_path": ""}
        """
        settings = get_settings()
        bucket_name = settings.get_bucket_name(agent_type, paper_type)

        if html_file_path is None:
            raise Exception("HTML文件路径不能为空")

        if paper_type == PaperTypeEnum.SYSTEM.value:
            object_name = f"{source}/{paper_id}/mindmap.html"
        else:
            object_name = f"{user_id}/{paper_id}/{result_id}/mindmap.html"

        content_type = "text/html"
        self.upload_file(bucket_name, html_file_path, object_name, content_type)

        return {
            "file_path": f"{bucket_name}/{object_name}"
        }

    def delete_task_results(
        self,
        agent_type: str,
        paper_type: str,
        paper_id: str,
        source: str,
        user_id: str
    ) -> bool:
        """删除任务结果文件

        Args:
            agent_type: 任务类型
            paper_type: 论文类型
            paper_id: 论文ID
            source: 论文来源
            user_id: 用户ID

        Returns:
            是否删除成功
        """
        settings = get_settings()
        bucket_name = settings.get_bucket_name(agent_type, paper_type)
        client = self._get_client()

        try:
            # 构建前缀
            if paper_type == PaperTypeEnum.SYSTEM.value:
                prefix = f"{source}/{paper_id}/"
            else:
                prefix = f"{user_id}/{paper_id}/"

            # 列出所有相关对象
            objects = client.list_objects(
                bucket_name,
                prefix=prefix,
                recursive=True
            )

            # 删除所有对象
            for obj in objects:
                client.remove_object(bucket_name, obj.object_name)
                logger.info(f"已删除: {bucket_name}/{obj.object_name}")

            return True
        except S3Error as e:
            error_msg = f"删除任务结果失败: {e.message if e.message else str(e)}"
            logger.error(error_msg)
            return False


@lru_cache()
def get_minio_service() -> MinIOService:
    """获取 MinIO 服务单例

    Returns:
        MinIOService: 服务实例
    """
    return MinIOService()
