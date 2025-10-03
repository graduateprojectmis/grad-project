"""
1. 清理原始文字資料，用於後續的文字處理與嵌入生成。
2. 輸入： 原始文字 (str)
3. 輸出： 清理後文字 (str)
"""

import re

def remove_headers_footers(text, header_patterns=None, footer_patterns=None):
    # 移除 header 與 footer 的預設模式
    if header_patterns is None: 
        header_patterns = [r'^.*Header.*$', r'^.*標題.*$'] 
    if footer_patterns is None:
        footer_patterns = [r'^.*Footer.*$', r'^.*頁尾.*$']

    for pattern in header_patterns + footer_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)

    return text.strip()

def remove_special_characters(text, keep_punctuations=True):
    if keep_punctuations:
        # 保留中文、英文、數字、常用標點符號
        pattern = r'[^A-Za-z0-9\u4e00-\u9fff\s\.,;:、\'\"\?\!\-\(\)（）]'
    else:
        pattern = r'[^A-Za-z0-9\u4e00-\u9fff\s]'
    
    text = re.sub(pattern, '', text)
    return text.strip()

def remove_repeated_substrings(text):
    # 將重複符號（如 ...... 或 !!!!）壓縮成單個符號
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    return text.strip()

def remove_extra_spaces(text):
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 多個空行壓縮成一個
    text = re.sub(r'[ \t]+', ' ', text)     # 多個空白或 tab 合併成一個
    return text.strip()

def preprocess_text(text):
    # 綜合清理函數
    text = remove_headers_footers(text)
    text = remove_special_characters(text)
    text = remove_repeated_substrings(text)
    text = remove_extra_spaces(text)
    return text
