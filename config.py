import os
from dotenv import load_dotenv

# 加载同目录下的 .env 文件
load_dotenv()

class Config:
    # DeepSeek 接口配置（兼容 OpenAI SDK 格式）
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # 嵌入模型配置（若无 OpenAI Key，RAG将自动降级为内置 Mock 模式以便基础运行）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")