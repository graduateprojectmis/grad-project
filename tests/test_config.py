"""
測試配置管理模組
"""
import pytest
from pathlib import Path
from app.config import get_settings, Settings


class TestSettings:
    """測試 Settings 類別"""
    
    def test_settings_singleton(self):
        """測試設定是單例模式"""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
    
    def test_default_values(self):
        """測試預設值"""
        settings = get_settings()
        
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.chunk_size == 600
        assert settings.chunk_overlap == 30
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.openai_embedding_model == "text-embedding-3-small"
        assert settings.openai_temperature == 0.3
    
    def test_data_directories_created(self):
        """測試資料目錄自動建立"""
        settings = get_settings()
        
        # 這些目錄應該在初始化時被建立
        assert settings.data_dir.exists()
        assert settings.output_dir.exists()
        assert settings.upload_dir.exists()
    
    def test_cors_origins(self):
        """測試 CORS 設定"""
        settings = get_settings()
        
        assert "http://localhost:3000" in settings.api_cors_origins
        assert "http://localhost:8080" in settings.api_cors_origins
    
    def test_chroma_db_settings(self):
        """測試 ChromaDB 設定"""
        settings = get_settings()
        
        assert settings.chroma_db_path == "./data/chroma_db"
        assert settings.chroma_collection_name == "airpods_manual"
    
    def test_log_settings(self):
        """測試日誌設定"""
        settings = get_settings()
        
        assert settings.log_level == "INFO"
        assert settings.log_file == "./logs/app.log"
        assert "%(asctime)s" in settings.log_format
