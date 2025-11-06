"""
測試資料模型
"""
import pytest
from pydantic import ValidationError
from app.models.schemas import (
    QuestionRequest,
    QuestionResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    DocumentChunk,
    DocumentWithEmbedding,
    APIKeyStatusResponse,
    SetAPIKeyRequest
)


class TestQuestionRequest:
    """測試問題請求模型"""
    
    def test_valid_request(self):
        """測試有效的請求"""
        request = QuestionRequest(
            question="如何配對 AirPods？",
            top_k=1
        )
        
        assert request.question == "如何配對 AirPods？"
        assert request.top_k == 1
    
    def test_default_top_k(self):
        """測試預設 top_k"""
        request = QuestionRequest(question="測試問題")
        assert request.top_k == 1
    
    def test_empty_question(self):
        """測試空問題"""
        with pytest.raises(ValidationError):
            QuestionRequest(question="", top_k=1)
    
    def test_invalid_top_k(self):
        """測試無效的 top_k"""
        with pytest.raises(ValidationError):
            QuestionRequest(question="測試", top_k=0)
        
        with pytest.raises(ValidationError):
            QuestionRequest(question="測試", top_k=11)


class TestQuestionResponse:
    """測試問題回應模型"""
    
    def test_valid_response(self):
        """測試有效的回應"""
        response = QuestionResponse(
            question="測試問題",
            answer="測試答案",
            status="success"
        )
        
        assert response.question == "測試問題"
        assert response.answer == "測試答案"
        assert response.status == "success"
        assert response.source_chunks is None
    
    def test_with_source_chunks(self):
        """測試包含來源片段的回應"""
        response = QuestionResponse(
            question="測試",
            answer="答案",
            status="success",
            source_chunks=["片段1", "片段2"]
        )
        
        assert len(response.source_chunks) == 2


class TestSearchRequest:
    """測試搜尋請求模型"""
    
    def test_valid_request(self):
        """測試有效的請求"""
        request = SearchRequest(
            query="搜尋關鍵字",
            n_results=5
        )
        
        assert request.query == "搜尋關鍵字"
        assert request.n_results == 5
    
    def test_default_n_results(self):
        """測試預設 n_results"""
        request = SearchRequest(query="測試")
        assert request.n_results == 1
    
    def test_invalid_n_results(self):
        """測試無效的 n_results"""
        with pytest.raises(ValidationError):
            SearchRequest(query="測試", n_results=0)
        
        with pytest.raises(ValidationError):
            SearchRequest(query="測試", n_results=11)


class TestHealthResponse:
    """測試健康檢查回應模型"""
    
    def test_valid_response(self):
        """測試有效的回應"""
        response = HealthResponse(
            status="healthy",
            message="API is running",
            chroma_db_count=100,
            version="2.0.0"
        )
        
        assert response.status == "healthy"
        assert response.message == "API is running"
        assert response.chroma_db_count == 100
        assert response.version == "2.0.0"


class TestDocumentModels:
    """測試文件相關模型"""
    
    def test_document_chunk(self):
        """測試文件片段"""
        chunk = DocumentChunk(
            chunk_text="這是一段文字",
            chunk_embedding=[0.1, 0.2, 0.3]
        )
        
        assert chunk.chunk_text == "這是一段文字"
        assert len(chunk.chunk_embedding) == 3
    
    def test_document_with_embedding(self):
        """測試包含嵌入的文件"""
        doc = DocumentWithEmbedding(
            title="測試標題",
            title_embedding=[0.1, 0.2],
            chunks=[
                DocumentChunk(
                    chunk_text="片段1",
                    chunk_embedding=[0.1, 0.2, 0.3]
                ),
                DocumentChunk(
                    chunk_text="片段2",
                    chunk_embedding=[0.4, 0.5, 0.6]
                )
            ]
        )
        
        assert doc.title == "測試標題"
        assert len(doc.chunks) == 2
        assert doc.chunks[0].chunk_text == "片段1"


class TestAPIKeyModels:
    """測試 API Key 相關模型"""
    
    def test_api_key_status_response(self):
        """測試 API Key 狀態回應"""
        response = APIKeyStatusResponse(
            exists=True,
            masked="sk-1234..."
        )
        
        assert response.exists is True
        assert response.masked == "sk-1234..."
    
    def test_set_api_key_request(self):
        """測試設定 API Key 請求"""
        request = SetAPIKeyRequest(api_key="sk-test-key")
        assert request.api_key == "sk-test-key"
    
    def test_empty_api_key(self):
        """測試空的 API Key"""
        with pytest.raises(ValidationError):
            SetAPIKeyRequest(api_key="")
