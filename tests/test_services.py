"""
測試服務層（使用 Mock）
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.database_service import DatabaseService
from app.core.exceptions import DatabaseError


class TestDatabaseService:
    """測試資料庫服務"""

    @patch("app.services.database_service.chromadb.PersistentClient")
    def test_initialization(self, mock_client, mock_db_path, mock_collection_name):
        """測試初始化"""
        # 設定 mock
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        # 初始化服務
        db_service = DatabaseService(
            db_path=mock_db_path, collection_name=mock_collection_name
        )

        # 驗證
        assert db_service.db_path == mock_db_path
        assert db_service.collection_name == mock_collection_name
        assert db_service.client is not None
        assert db_service.collection is not None

    @patch("app.services.database_service.chromadb.PersistentClient")
    def test_insert(self, mock_client, mock_db_path):
        """測試插入資料"""
        # 設定 mock
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        db_service = DatabaseService(db_path=mock_db_path)

        # 測試資料
        ids = ["id1", "id2"]
        documents = ["doc1", "doc2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"type": "test"}, {"type": "test"}]

        # 執行插入
        db_service.insert(ids, documents, embeddings, metadatas)

        # 驗證 mock 被呼叫
        mock_collection.add.assert_called_once_with(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

    @patch("app.services.database_service.chromadb.PersistentClient")
    def test_query(self, mock_client, mock_db_path):
        """測試查詢"""
        # 設定 mock
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        db_service = DatabaseService(db_path=mock_db_path)

        # 執行查詢
        query_embedding = [0.1, 0.2, 0.3]
        documents, distances = db_service.query(query_embedding, n_results=2)

        # 驗證結果
        assert documents == ["doc1", "doc2"]
        assert distances == [0.1, 0.2]

        # 驗證 mock 被呼叫
        mock_collection.query.assert_called_once()

    @patch("app.services.database_service.chromadb.PersistentClient")
    def test_count(self, mock_client, mock_db_path):
        """測試計數"""
        # 設定 mock
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        db_service = DatabaseService(db_path=mock_db_path)

        # 執行計數
        count = db_service.count()

        # 驗證結果
        assert count == 42

    @patch("app.services.database_service.chromadb.PersistentClient")
    def test_clear(self, mock_client, mock_db_path, mock_collection_name):
        """測試清空"""
        # 設定 mock
        mock_collection = Mock()
        mock_instance = mock_client.return_value
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_instance.create_collection.return_value = mock_collection

        db_service = DatabaseService(
            db_path=mock_db_path, collection_name=mock_collection_name
        )

        # 執行清空
        db_service.clear()

        # 驗證 mock 被呼叫
        mock_instance.delete_collection.assert_called_once_with(
            name=mock_collection_name
        )
        mock_instance.create_collection.assert_called_once_with(
            name=mock_collection_name
        )

    @patch("app.services.database_service.chromadb.PersistentClient")
    def test_error_handling(self, mock_client, mock_db_path):
        """測試錯誤處理"""
        # 設定 mock 拋出例外
        mock_client.side_effect = Exception("Connection failed")

        # 應該拋出 DatabaseError
        with pytest.raises(DatabaseError) as exc_info:
            DatabaseService(db_path=mock_db_path)

        assert "資料庫初始化失敗" in str(exc_info.value)


class TestEmbeddingServiceMock:
    """測試嵌入服務（使用 Mock）"""

    @patch("app.services.embedding_service.OpenAI")
    def test_openai_embedding(self, mock_openai_class):
        """測試 OpenAI 嵌入生成"""
        from app.services.embedding_service import OpenAIEmbeddingService

        # 設定 mock 回應
        mock_response = Mock()
        mock_item1 = Mock()
        mock_item1.embedding = [0.1, 0.2, 0.3]
        mock_item2 = Mock()
        mock_item2.embedding = [0.4, 0.5, 0.6]
        mock_response.data = [mock_item1, mock_item2]

        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # 初始化服務
        service = OpenAIEmbeddingService(api_key="test-key")

        # 生成嵌入
        result = service.generate_embedding(["text1", "text2"])

        # 驗證結果
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    def test_embedding_service_factory(self):
        """測試嵌入服務工廠"""
        from app.services.embedding_service import EmbeddingService

        # 測試 OpenAI
        with patch("app.services.embedding_service.OpenAIEmbeddingService"):
            service = EmbeddingService(provider="openai", api_key="test-key")
            assert service.provider == "openai"

        # 測試無效的提供者
        with pytest.raises(ValueError):
            EmbeddingService(provider="invalid")


class TestLLMServiceMock:
    """測試 LLM 服務（使用 Mock）"""

    @patch("app.services.llm_service.OpenAI")
    def test_generate_answer(self, mock_openai_class):
        """測試生成答案"""
        from app.services.llm_service import LLMService

        # 設定 mock 回應
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "這是測試答案"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # 初始化服務
        service = LLMService(api_key="test-key")

        # 生成答案
        answer = service.generate_answer(question="測試問題", context="測試上下文")

        # 驗證結果
        assert answer == "這是測試答案"

        # 驗證 mock 被呼叫
        mock_client.chat.completions.create.assert_called_once()

    @patch("app.services.llm_service.OpenAI")
    def test_generate_summary(self, mock_openai_class):
        """測試生成摘要"""
        from app.services.llm_service import LLMService

        # 設定 mock 回應
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "這是摘要"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # 初始化服務
        service = LLMService(api_key="test-key")

        # 生成摘要
        summary = service.generate_summary("這是一段很長的文字" * 100)

        # 驗證結果
        assert summary == "這是摘要"


class TestImageAnnotationService:
    """測試圖像標註服務"""

    def test_detected_object_creation(self):
        """測試 DetectedObject 建立"""
        from app.services.annotating_service import DetectedObject

        box_2d = [100, 200, 300, 400]
        label = "test_button"

        obj = DetectedObject(box_2d=box_2d, label=label)

        assert obj.box_2d == box_2d
        assert obj.label == label
        assert obj.to_dict() == {"box_2d": box_2d, "label": label}

    @patch("app.services.annotating_service.genai.Client")
    def test_initialization(self, mock_client):
        """測試服務初始化"""
        from app.services.annotating_service import ImageAnnotationService

        service = ImageAnnotationService(api_key="test-key")

        assert service.api_key == "test-key"
        assert service.model == "gemini-2.0-flash-exp"
        assert service.max_image_size == (1024, 1024)
        mock_client.assert_called_once_with(api_key="test-key")

    def test_parse_json_response(self):
        """測試 JSON 回應解析"""
        from app.services.annotating_service import ImageAnnotationService

        service = ImageAnnotationService(api_key="test-key")

        # 測試帶有 markdown 標記的回應
        json_with_markdown = """```json
[{"box_2d": [100, 200, 300, 400], "label": "button"}]
```"""
        result = service._parse_json_response(json_with_markdown)
        assert "```" not in result
        assert "[{" in result

        # 測試純 JSON 回應
        pure_json = '[{"box_2d": [100, 200, 300, 400], "label": "button"}]'
        result = service._parse_json_response(pure_json)
        assert result == pure_json

    @patch("app.services.annotating_service.Image.open")
    def test_load_and_resize_image(self, mock_open):
        """測試載入和調整圖像大小"""
        from app.services.annotating_service import ImageAnnotationService
        from PIL import Image

        # 建立 mock 圖像
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (2048, 2048)
        mock_open.return_value = mock_image

        service = ImageAnnotationService(api_key="test-key")
        result = service._load_and_resize_image("test.png")

        # 驗證
        mock_open.assert_called_once_with("test.png")
        mock_image.thumbnail.assert_called_once()

    @patch("app.services.annotating_service.genai.Client")
    @patch("app.services.annotating_service.Image.open")
    def test_detect_objects(self, mock_open, mock_client_class):
        """測試物件偵測"""
        from app.services.annotating_service import ImageAnnotationService
        from PIL import Image

        # 建立 mock 圖像
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (1024, 1024)
        mock_open.return_value = mock_image

        # 建立 mock API 回應
        mock_response = Mock()
        mock_response.text = """```json
[
    {"box_2d": [100, 200, 300, 400], "label": "play_button"},
    {"box_2d": [500, 600, 700, 800], "label": "pause_button"}
]
```"""

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        # 執行偵測
        service = ImageAnnotationService(api_key="test-key")
        detected_objects = service.detect_objects("test.png", "button")

        # 驗證結果
        assert len(detected_objects) == 2
        assert detected_objects[0].label == "play_button"
        assert detected_objects[1].label == "pause_button"
        assert detected_objects[0].box_2d == [100, 200, 300, 400]

    @patch("app.services.annotating_service.genai.Client")
    @patch("app.services.annotating_service.Image.open")
    def test_get_detection_summary(self, mock_open, mock_client_class):
        """測試獲取偵測摘要"""
        from app.services.annotating_service import ImageAnnotationService
        from PIL import Image

        # 建立 mock 圖像
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (1024, 1024)
        mock_open.return_value = mock_image

        # 建立 mock API 回應
        mock_response = Mock()
        mock_response.text = (
            '[{"box_2d": [100, 200, 300, 400], "label": "test_button"}]'
        )

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        # 執行
        service = ImageAnnotationService(api_key="test-key")
        summary = service.get_detection_summary("test.png", "button")

        # 驗證
        assert summary["image_path"] == "test.png"
        assert summary["target_item"] == "button"
        assert summary["total_detected"] == 1
        assert len(summary["objects"]) == 1
        assert summary["objects"][0]["label"] == "test_button"
