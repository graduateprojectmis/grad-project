"""
測試配置和共用 fixtures
"""

import pytest
import os
import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def settings():
    """測試設定"""
    from app.config import get_settings

    return get_settings()


@pytest.fixture
def test_api_key():
    """測試用的 API Key"""
    return os.getenv("OPENAI_API_KEY", "test-key-12345")


@pytest.fixture
def sample_texts():
    """測試用的文字範例"""
    return [
        "如何配對 AirPods 到 iPhone？",
        "AirPods 的電池壽命有多長？",
        "怎麼重置 AirPods？",
    ]


@pytest.fixture
def sample_embeddings():
    """測試用的嵌入向量（假資料）"""
    return [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]  # OpenAI embedding dimension


@pytest.fixture
def mock_db_path(tmp_path):
    """臨時資料庫路徑"""
    return str(tmp_path / "test_chroma_db")


@pytest.fixture
def mock_collection_name():
    """測試用的集合名稱"""
    return "test_collection"
