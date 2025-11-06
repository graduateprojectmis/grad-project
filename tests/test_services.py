"""
測試服務層（使用 Mock）
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.database_service import DatabaseService
from app.core.exceptions import DatabaseError


class TestDatabaseService:
    """測試資料庫服務"""
    
    @patch('app.services.database_service.chromadb.PersistentClient')
    def test_initialization(self, mock_client, mock_db_path, mock_collection_name):
        """測試初始化"""
        # 設定 mock
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # 初始化服務
        db_service = DatabaseService(
            db_path=mock_db_path,
            collection_name=mock_collection_name
        )
        
        # 驗證
        assert db_service.db_path == mock_db_path
        assert db_service.collection_name == mock_collection_name
        assert db_service.client is not None
        assert db_service.collection is not None
    
    @patch('app.services.database_service.chromadb.PersistentClient')
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
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    @patch('app.services.database_service.chromadb.PersistentClient')
    def test_query(self, mock_client, mock_db_path):
        """測試查詢"""
        # 設定 mock
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]]
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
    
    @patch('app.services.database_service.chromadb.PersistentClient')
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
    
    @patch('app.services.database_service.chromadb.PersistentClient')
    def test_clear(self, mock_client, mock_db_path, mock_collection_name):
        """測試清空"""
        # 設定 mock
        mock_collection = Mock()
        mock_instance = mock_client.return_value
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_instance.create_collection.return_value = mock_collection
        
        db_service = DatabaseService(
            db_path=mock_db_path,
            collection_name=mock_collection_name
        )
        
        # 執行清空
        db_service.clear()
        
        # 驗證 mock 被呼叫
        mock_instance.delete_collection.assert_called_once_with(name=mock_collection_name)
        mock_instance.create_collection.assert_called_once_with(name=mock_collection_name)
    
    @patch('app.services.database_service.chromadb.PersistentClient')
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
    
    @patch('app.services.embedding_service.openai.Embedding.create')
    def test_openai_embedding(self, mock_create):
        """測試 OpenAI 嵌入生成"""
        from app.services.embedding_service import OpenAIEmbeddingService
        
        # 設定 mock 回應
        mock_create.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }
        
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
        with patch('app.services.embedding_service.OpenAIEmbeddingService'):
            service = EmbeddingService(provider="openai", api_key="test-key")
            assert service.provider == "openai"
        
        # 測試無效的提供者
        with pytest.raises(ValueError):
            EmbeddingService(provider="invalid")


class TestLLMServiceMock:
    """測試 LLM 服務（使用 Mock）"""
    
    @patch('app.services.llm_service.openai.ChatCompletion.create')
    def test_generate_answer(self, mock_create):
        """測試生成答案"""
        from app.services.llm_service import LLMService
        
        # 設定 mock 回應
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "這是測試答案"
        mock_create.return_value = mock_response
        
        # 初始化服務
        service = LLMService(api_key="test-key")
        
        # 生成答案
        answer = service.generate_answer(
            question="測試問題",
            context="測試上下文"
        )
        
        # 驗證結果
        assert answer == "這是測試答案"
        
        # 驗證 mock 被呼叫
        mock_create.assert_called_once()
    
    @patch('app.services.llm_service.openai.ChatCompletion.create')
    def test_generate_summary(self, mock_create):
        """測試生成摘要"""
        from app.services.llm_service import LLMService
        
        # 設定 mock 回應
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "這是摘要"
        mock_create.return_value = mock_response
        
        # 初始化服務
        service = LLMService(api_key="test-key")
        
        # 生成摘要
        summary = service.generate_summary("這是一段很長的文字" * 100)
        
        # 驗證結果
        assert summary == "這是摘要"
