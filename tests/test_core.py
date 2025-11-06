"""
測試核心功能模組
"""
import pytest
import logging
from pathlib import Path
from app.core.logger import setup_logger, get_logger
from app.core.exceptions import (
    AppException,
    DatabaseError,
    EmbeddingError,
    APIKeyError,
    ValidationError
)


class TestLogger:
    """測試日誌系統"""
    
    def test_setup_logger(self, tmp_path):
        """測試日誌設定"""
        log_file = tmp_path / "test.log"
        logger = setup_logger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 2  # 控制台 + 檔案
    
    def test_get_logger(self):
        """測試獲取 Logger"""
        logger = get_logger("test_module")
        assert logger.name == "test_module"
        assert isinstance(logger, logging.Logger)
    
    def test_logger_file_creation(self, tmp_path):
        """測試日誌檔案建立"""
        log_file = tmp_path / "subdir" / "test.log"
        logger = setup_logger(
            name="file_test",
            log_file=str(log_file)
        )
        
        # 寫入日誌
        logger.info("Test message")
        
        # 確認檔案存在
        assert log_file.exists()
        
        # 確認內容
        content = log_file.read_text()
        assert "Test message" in content
    
    def test_logger_levels(self, tmp_path):
        """測試不同日誌級別"""
        log_file = tmp_path / "level_test.log"
        logger = setup_logger(
            name="level_test",
            level="WARNING",
            log_file=str(log_file)
        )
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        content = log_file.read_text()
        
        # DEBUG 和 INFO 不應該被記錄
        assert "Debug message" not in content
        assert "Info message" not in content
        
        # WARNING 和 ERROR 應該被記錄
        assert "Warning message" in content
        assert "Error message" in content


class TestExceptions:
    """測試自定義例外"""
    
    def test_app_exception(self):
        """測試基礎例外"""
        exc = AppException("Test error", {"key": "value"})
        
        assert exc.message == "Test error"
        assert exc.details == {"key": "value"}
        assert str(exc) == "Test error"
    
    def test_database_error(self):
        """測試資料庫錯誤"""
        exc = DatabaseError("Database connection failed")
        
        assert isinstance(exc, AppException)
        assert exc.message == "Database connection failed"
    
    def test_embedding_error(self):
        """測試嵌入錯誤"""
        exc = EmbeddingError("Embedding generation failed")
        
        assert isinstance(exc, AppException)
        assert exc.message == "Embedding generation failed"
    
    def test_api_key_error(self):
        """測試 API Key 錯誤"""
        exc = APIKeyError("API Key not set")
        
        assert isinstance(exc, AppException)
        assert exc.message == "API Key not set"
    
    def test_validation_error(self):
        """測試驗證錯誤"""
        exc = ValidationError("Invalid input", {"field": "email"})
        
        assert isinstance(exc, AppException)
        assert exc.message == "Invalid input"
        assert exc.details["field"] == "email"
    
    def test_exception_inheritance(self):
        """測試例外繼承關係"""
        assert issubclass(DatabaseError, AppException)
        assert issubclass(EmbeddingError, AppException)
        assert issubclass(APIKeyError, AppException)
        assert issubclass(ValidationError, AppException)
