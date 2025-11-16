"""
測試 API 端點
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from app.api.main import app


@pytest.fixture
def client():
    """測試客戶端"""
    return TestClient(app)


class TestHealthEndpoint:
    """測試健康檢查端點"""

    def test_health_check(self, client):
        """測試健康檢查"""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "message" in data
        assert "chroma_db_count" in data
        assert "version" in data


class TestRootEndpoint:
    """測試根路徑端點"""

    def test_root(self, client):
        """測試根路徑"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"


class TestAskEndpoint:
    """測試問答端點"""

    @patch("app.api.main.embedding_service")
    @patch("app.api.main.db_service")
    @patch("app.api.main.llm_service")
    def test_ask_question_success(self, mock_llm, mock_db, mock_embed, client):
        """測試成功的問答"""
        # 設定 mocks
        mock_embed.generate_embedding.return_value = [[0.1] * 1536]
        mock_db.query.return_value = (["相關文件內容"], [0.1])
        mock_llm.generate_answer.return_value = "這是答案"

        # 發送請求
        response = client.post("/api/ask", json={"question": "測試問題", "top_k": 1})

        # 驗證回應
        assert response.status_code == 200
        data = response.json()

        assert data["question"] == "測試問題"
        assert data["answer"] == "這是答案"
        assert data["status"] == "success"

    def test_ask_empty_question(self, client):
        """測試空問題"""
        response = client.post("/api/ask", json={"question": "", "top_k": 1})

        # Pydantic validation 回應 422
        assert response.status_code == 422

    def test_ask_invalid_top_k(self, client):
        """測試無效的 top_k"""
        response = client.post("/api/ask", json={"question": "測試", "top_k": 0})

        assert response.status_code == 422  # Validation error


class TestSearchEndpoint:
    """測試搜尋端點"""

    @patch("app.api.main.embedding_service")
    @patch("app.api.main.db_service")
    def test_search_success(self, mock_db, mock_embed, client):
        """測試成功的搜尋"""
        # 設定 mocks
        mock_embed.generate_embedding.return_value = [[0.1] * 1536]
        mock_db.query.return_value = (["結果1", "結果2"], [0.1, 0.2])

        # 發送請求
        response = client.post(
            "/api/search", json={"query": "搜尋測試", "n_results": 2}
        )

        # 驗證回應
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "搜尋測試"
        assert len(data["results"]) == 2
        assert data["status"] == "success"

    def test_search_empty_query(self, client):
        """測試空查詢"""
        response = client.post("/api/search", json={"query": "", "n_results": 1})

        # Pydantic validation 回應 422
        assert response.status_code == 422


class TestAdminEndpoints:
    """測試管理端點"""

    @patch("app.api.main.settings")
    def test_get_api_key_status(self, mock_settings, client):
        """測試獲取 API Key 狀態"""
        mock_settings.openai_api_key = "sk-test1234567890"

        response = client.get("/api/admin/api-key/status")

        # 可能需要 token，本測試接受 200 或 401
        assert response.status_code in [200, 401]

        if response.status_code == 200:
            data = response.json()
            assert "exists" in data

    @patch("app.api.main.set_key")
    @patch("app.api.main.settings")
    def test_set_api_key(self, mock_settings, mock_set_key, client):
        """測試設定 API Key"""
        response = client.post(
            "/api/admin/api-key", json={"api_key": "sk-new-test-key"}
        )

        # 本機請求應該成功
        assert response.status_code in [200, 401]  # 可能需要 token

    def test_set_invalid_api_key(self, client):
        """測試設定無效的 API Key"""
        response = client.post("/api/admin/api-key", json={"api_key": "invalid-key"})

        # 應該返回錯誤（格式錯誤或需要授權）
        assert response.status_code in [400, 401]
