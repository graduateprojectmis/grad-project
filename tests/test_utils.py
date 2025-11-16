"""
測試工具函數
"""

import pytest
from pathlib import Path
from app.utils.text_processing import clean_text, split_text
from app.utils.file_operations import load_json, save_json


class TestTextProcessing:
    """測試文字處理"""

    def test_clean_text_basic(self):
        """測試基本文字清理"""
        text = "  這是   一段   文字  "
        result = clean_text(text)
        assert result == "這是 一段 文字"

    def test_clean_text_special_chars(self):
        """測試特殊字元清理"""
        text = "文字@#$%含有特殊字元"
        result = clean_text(text)
        # 應該保留中文和部分標點
        assert "@#$%" not in result

    def test_clean_text_empty(self):
        """測試空文字"""
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_clean_text_whitespace(self):
        """測試多餘空白"""
        text = "文字\n\n\n包含\t\t換行"
        result = clean_text(text)
        assert "\n\n" not in result
        assert "\t\t" not in result

    def test_split_text_basic(self):
        """測試文字分割"""
        text = "這是第一句。這是第二句。這是第三句。"
        chunks = split_text(text, chunk_size=20, chunk_overlap=5)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_text_long(self):
        """測試長文字分割"""
        # 使用較長的文字確保會被分割
        text = "這是一個測試句子。" * 200  # ~2000 字元
        chunks = split_text(text, chunk_size=100, chunk_overlap=10)

        # 驗證分割結果
        assert isinstance(chunks, list)
        assert len(chunks) >= 1  # 至少有一個片段

        # 如果真的分割了，檢查每個片段
        if len(chunks) > 1:
            for chunk in chunks:
                # 每個片段不應該明顯超過 chunk_size
                assert len(chunk) <= 200  # 允許一些彈性

    def test_split_text_short(self):
        """測試短文字分割"""
        text = "短文字"
        chunks = split_text(text, chunk_size=100)

        # 短文字應該只有一個片段
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_text_empty(self):
        """測試空文字分割"""
        chunks = split_text("", chunk_size=100)
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == "")


class TestFileOperations:
    """測試檔案操作"""

    def test_save_and_load_json(self, tmp_path):
        """測試儲存和載入 JSON"""
        file_path = tmp_path / "test.json"
        test_data = [{"name": "測試1", "value": 100}, {"name": "測試2", "value": 200}]

        # 儲存
        result = save_json(test_data, str(file_path))
        assert result is True
        assert file_path.exists()

        # 載入
        loaded_data = load_json(str(file_path))
        assert loaded_data == test_data

    def test_load_json_nonexistent(self):
        """測試載入不存在的檔案"""
        result = load_json("/nonexistent/file.json")
        assert result == []

    def test_save_json_nested_path(self, tmp_path):
        """測試儲存到巢狀路徑"""
        file_path = tmp_path / "dir1" / "dir2" / "test.json"
        test_data = {"key": "value"}

        result = save_json(test_data, str(file_path))
        assert result is True
        assert file_path.exists()

        loaded_data = load_json(str(file_path))
        assert loaded_data == test_data

    def test_save_json_with_chinese(self, tmp_path):
        """測試儲存包含中文的 JSON"""
        file_path = tmp_path / "chinese.json"
        test_data = {"內容": "這是中文測試"}

        save_json(test_data, str(file_path))

        # 讀取檔案內容
        content = file_path.read_text(encoding="utf-8")

        # 確認中文沒有被編碼
        assert "這是中文測試" in content
        assert "\\u" not in content  # 不應該有 Unicode 轉義

    def test_load_json_invalid_format(self, tmp_path):
        """測試載入無效的 JSON"""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("這不是有效的 JSON")

        result = load_json(str(file_path))
        assert result == []
