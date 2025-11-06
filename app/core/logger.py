"""
日誌系統
提供統一的日誌記錄功能
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "app",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    設定日誌記錄器
    
    Args:
        name: Logger 名稱
        level: 日誌級別
        log_file: 日誌檔案路徑
        log_format: 日誌格式
        
    Returns:
        Logger 實例
    """
    logger = logging.getLogger(name)
    
    # 避免重複設定
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # 預設格式
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 檔案處理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    獲取 Logger 實例
    
    Args:
        name: Logger 名稱
        
    Returns:
        Logger 實例
    """
    return logging.getLogger(name)
