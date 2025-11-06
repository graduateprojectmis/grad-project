"""
檔案操作工具
"""
import json
from pathlib import Path
from typing import Any, List, Dict

from app.core.logger import get_logger

logger = get_logger(__name__)


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """
    載入 JSON 檔案
    
    Args:
        file_path: 檔案路徑
        
    Returns:
        JSON 資料
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"檔案不存在：{file_path}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"成功載入 JSON 檔案：{file_path}")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 格式錯誤：{e}")
        return []
    except Exception as e:
        logger.error(f"載入檔案時發生錯誤：{e}")
        return []


def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    儲存為 JSON 檔案
    
    Args:
        data: 要儲存的資料
        file_path: 檔案路徑
        indent: 縮排空格數
        
    Returns:
        是否成功
    """
    try:
        path = Path(file_path)
        
        # 確保目錄存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        logger.info(f"成功儲存 JSON 檔案：{file_path}")
        return True
        
    except Exception as e:
        logger.error(f"儲存檔案時發生錯誤：{e}")
        return False
