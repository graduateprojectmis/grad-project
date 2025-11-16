"""
應用配置設定
使用 Pydantic Settings 管理環境變數
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """應用程式設定"""

    # API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    admin_token: Optional[str] = None

    # OpenAI 設定
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.3

    # ChromaDB 設定
    chroma_db_path: str = "./data/chroma_db"
    chroma_collection_name: str = "airpods_manual"

    # 文本處理設定
    chunk_size: int = 600
    chunk_overlap: int = 30

    # API 設定
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8080",
        "http://localhost:5500",
    ]

    # 日誌設定
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 資料路徑
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./data/output")
    upload_dir: Path = Path("./data/uploads")

    # 專案路徑
    project_root: Path = Path(__file__).parent.parent.parent

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 確保目錄存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """獲取應用設定（單例模式）"""
    return Settings()
