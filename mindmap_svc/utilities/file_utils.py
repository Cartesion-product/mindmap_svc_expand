import os
import re
import uuid
import shutil
import platform
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Union

from minio import Minio
from minio.error import S3Error


class MinioUploader:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        """
        初始化 MinIO 客户端

        :param endpoint: MinIO 服务地址 (e.g., "play.min.io:9000")
        :param access_key: MinIO 访问密钥
        :param secret_key: MinIO 秘密密钥
        :param secure: 是否使用 HTTPS (默认 False)
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        # 文件ID到对象名的映射缓存（UUID -> UUID.ext）
        self._file_mapping = {}

    def _generate_file_id(self) -> str:
        """生成唯一文件 ID (UUID)"""
        return str(uuid.uuid4())

    def _get_file_extension(self, file_path: str) -> str:
        """获取文件扩展名"""
        return os.path.splitext(file_path)[1].lower() if file_path else ""

    def upload_file(self, file_path: str, bucket_name: str, custom_name: str = None) -> str:
        """
        上传单个文件到 MinIO

        :param file_path: 本地文件路径
        :param bucket_name: MinIO 存储桶名称
        :param custom_name: 自定义文件名（不包含扩展名），为None则使用UUID
        :return: 生成的文件 ID (UUID 或自定义名称)
        """
        # 1. 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 2. 生成文件 ID 和对象名
        file_ext = self._get_file_extension(file_path)
        
        if custom_name:
            # 使用自定义名称
            file_id = custom_name
            object_name = f"{custom_name}{file_ext}"
        else:
            # 使用UUID
            file_id = self._generate_file_id()
            object_name = f"{file_id}{file_ext}"
        
        # 保存映射关系
        self._file_mapping[file_id] = object_name

        # 3. 创建存储桶 (如果不存在)
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

        # 4. 上传文件
        try:
            self.client.fput_object(
                bucket_name,
                object_name,
                file_path
            )
            return file_id
        except S3Error as e:
            raise Exception(f"MinIO 上传失败: {str(e)}") from e

    def batch_upload_files(self, file_paths: List[str], bucket_name: str) -> List[str]:
        """
        批量上传文件到 MinIO

        :param file_paths: 本地文件路径列表
        :param bucket_name: MinIO 存储桶名称
        :return: 生成的文件 ID 列表
        """
        if not file_paths:
            return []

        # 创建存储桶 (如果不存在)
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

        file_ids = []
        for file_path in file_paths:
            try:
                file_id = self.upload_file(file_path, bucket_name)
                file_ids.append(file_id)
            except Exception as e:
                # 记录错误但继续处理其他文件
                print(f"上传失败 {file_path}: {str(e)}")
                # 可选：将失败文件添加到错误列表
                # failed_files.append(file_path)

        return file_ids

    def get_file_url(self, bucket_name: str, file_id: str, expires: int = 3600) -> str:
        """
        获取文件的临时访问 URL

        :param bucket_name: 存储桶名称
        :param file_id: 文件 ID (UUID)
        :param expires: URL 过期时间(秒), 默认 1 小时
        :return: 临时 URL
        """
        # 从映射中获取对象名
        object_name = self._file_mapping.get(file_id)
        if not object_name:
            # 如果映射不存在，尝试使用常规命名（仅适用于PDF）
            object_name = f"{file_id}.pdf"
        
        # 将秒数转换为timedelta对象
        expires_delta = timedelta(seconds=expires)
        
        return self.client.get_presigned_url(
            "GET",
            bucket_name,
            object_name,
            expires=expires_delta
        )


# 使用示例
if __name__ == "__main__":
    # 配置 MinIO 信息 (实际使用时从环境变量获取)
    MINIO_ENDPOINT = "your-minio-server:9000"
    MINIO_ACCESS_KEY = "YOUR_ACCESS_KEY"
    MINIO_SECRET_KEY = "YOUR_SECRET_KEY"
    BUCKET_NAME = "your-bucket-name"

    # 初始化上传器
    uploader = MinioUploader(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # 使用 HTTP
    )

    # 单文件上传
    try:
        file_id = uploader.upload_file("local_file.jpg", BUCKET_NAME)
        print(f"单文件上传成功! 文件ID: {file_id}")
        print(f"访问链接: {uploader.get_file_url(BUCKET_NAME, file_id)}")
    except Exception as e:
        print(f"单文件上传失败: {str(e)}")

    # 批量上传
    try:
        file_paths = ["file1.jpg", "file2.pdf", "file3.png"]
        file_ids = uploader.batch_upload_files(file_paths, BUCKET_NAME)
        print(f"批量上传成功! 文件ID列表: {file_ids}")
    except Exception as e:
        print(f"批量上传失败: {str(e)}")
