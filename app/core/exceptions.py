"""
自定義例外類別
"""


class AppException(Exception):
    """應用程式基礎例外"""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DatabaseError(AppException):
    """資料庫相關錯誤"""

    pass


class EmbeddingError(AppException):
    """向量嵌入相關錯誤"""

    pass


class APIKeyError(AppException):
    """API Key 相關錯誤"""

    pass


class ValidationError(AppException):
    """資料驗證錯誤"""

    pass
