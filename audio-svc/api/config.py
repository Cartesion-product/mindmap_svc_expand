"""
API Configuration Management
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """API配置类"""

    # 模型配置
    model_path: str = os.getenv(
        "MODEL_PATH",
        "pretrained_models/SoulX-Podcast-1.7B"
    )
    llm_engine: str = os.getenv("LLM_ENGINE", "hf")  # hf or vllm
    fixed_prompt_audio_paths = [
        "man.wav",
        "woman.WAV",
    ]

    # 固定参考文本（必须与音频一一对应）
    fixed_prompt_texts = [
        "您好，我是小渊。小渊论文精讲时刻，欢迎您！作为您的专属学术同行者，我将与您一起探索前沿新知的无限可能。让我们即刻开启学术慧思，携手畅快破局，共驶识渊彼岸吧！",
        "您好，我是小渊。小渊论文精讲时刻，欢迎您！作为您的专属学术同行者，我将与您一起探索前沿新知的无限可能。让我们即刻开启学术慧思，携手畅快破局，共驶识渊彼岸吧！",
    ]

    def validate_llm_engine(self):
        """验证LLM引擎配置"""
        if self.llm_engine not in ["hf", "vllm"]:
            raise ValueError(f"Invalid llm_engine: {self.llm_engine}. Must be 'hf' or 'vllm'")

        # 如果选择vllm，检查是否安装
        if self.llm_engine == "vllm":
            try:
                import vllm
            except ImportError:
                import logging
                logging.warning("vLLM not installed, falling back to HuggingFace engine")
                self.llm_engine = "hf"
    fp16_flow: bool = os.getenv("FP16_FLOW", "false").lower() == "true"

    # 服务配置
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8065"))
    reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"

    # 文件配置
    temp_dir: Path = Path("api/temp")
    output_dir: Path = Path("api/outputs")
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    file_cleanup_minutes: int = 30  # 文件过期时间（分钟）

    # 并发控制
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))

    # 默认生成参数
    default_seed: int = 1988
    default_temperature: float = 0.6
    default_top_k: int = 100
    default_top_p: float = 0.9

    def __post_init__(self):
        """确保目录存在并验证配置"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validate_llm_engine()


# 全局配置实例
config = APIConfig()
