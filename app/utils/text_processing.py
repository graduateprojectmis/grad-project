"""
文字處理工具
"""

import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import get_logger
from app.config import get_settings

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    清理文字

    Args:
        text: 原始文字

    Returns:
        清理後的文字
    """
    if not text:
        return ""

    # 移除多餘空白
    text = re.sub(r"\s+", " ", text)

    # 移除特殊字元（保留中文、英文、數字、標點符號）
    text = re.sub(r"[^\w\s\u4e00-\u9fff.,!?;:，。！？；：、]", "", text)

    # 去除首尾空白
    text = text.strip()

    return text


def split_text(
    text: str, chunk_size: int = None, chunk_overlap: int = None
) -> List[str]:
    """
    將文字分割成片段

    Args:
        text: 原始文字
        chunk_size: 片段大小
        chunk_overlap: 片段重疊大小

    Returns:
        文字片段列表
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    logger.debug(f"分割文字，片段大小：{chunk_size}，重疊：{chunk_overlap}")

    # 初始化文本切分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["。", "！", "？", "\n", "，", " "],
    )

    # 清理文字
    cleaned_text = clean_text(text)

    # 分割文字
    chunks = text_splitter.split_text(cleaned_text)

    logger.debug(f"文字分割完成，共 {len(chunks)} 個片段")

    return chunks
