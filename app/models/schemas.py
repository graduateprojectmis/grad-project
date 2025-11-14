"""
資料模型定義
使用 Pydantic 定義 API 的請求和回應格式
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """問題請求"""
    question: str = Field(..., min_length=1, description="使用者問題")
    top_k: int = Field(default=1, ge=1, le=10, description="返回結果數量")
    
    model_config = {"json_schema_extra": {
        "examples": [{
            "question": "如何配對 AirPods？",
            "top_k": 1
        }]
    }}


class QuestionResponse(BaseModel):
    """問題回應"""
    question: str = Field(..., description="使用者問題")
    answer: str = Field(..., description="AI 生成的答案")
    status: str = Field(default="success", description="狀態")
    source_chunks: Optional[List[str]] = Field(default=None, description="來源文件片段")


class SearchRequest(BaseModel):
    """搜尋請求"""
    query: str = Field(..., min_length=1, description="搜尋查詢")
    n_results: int = Field(default=1, ge=1, le=10, description="返回結果數量")


class SearchResponse(BaseModel):
    """搜尋回應"""
    query: str = Field(..., description="搜尋查詢")
    results: List[str] = Field(..., description="搜尋結果")
    status: str = Field(default="success", description="狀態")


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str = Field(..., description="服務狀態")
    message: str = Field(..., description="狀態訊息")
    chroma_db_count: int = Field(..., description="資料庫文件數量")
    version: str = Field(..., description="版本號")


class EmbeddingRequest(BaseModel):
    """嵌入向量請求"""
    texts: List[str] = Field(..., min_items=1, description="待嵌入的文字列表")
    model: Optional[str] = Field(default=None, description="使用的模型")


class EmbeddingResponse(BaseModel):
    """嵌入向量回應"""
    embeddings: List[List[float]] = Field(..., description="嵌入向量列表")
    model: str = Field(..., description="使用的模型")
    status: str = Field(default="success", description="狀態")


class DocumentChunk(BaseModel):
    """文件片段"""
    chunk_text: str = Field(..., description="片段文字")
    chunk_embedding: List[float] = Field(..., description="片段嵌入向量")


class DocumentWithEmbedding(BaseModel):
    """包含嵌入向量的文件"""
    title: str = Field(..., description="文件標題")
    title_embedding: List[float] = Field(..., description="標題嵌入向量")
    chunks: List[DocumentChunk] = Field(..., description="文件片段列表")
    
    
class APIKeyStatusResponse(BaseModel):
    """API Key 狀態回應"""
    exists: bool = Field(..., description="是否存在 API Key")
    masked: str = Field(..., description="遮罩後的 API Key")


class SetAPIKeyRequest(BaseModel):
    """設定 API Key 請求"""
    api_key: str = Field(..., min_length=1, description="API Key")


class ImageAnnotationRequest(BaseModel):
    """圖片標註請求"""
    target_item: str = Field(default="objects", description="目標物件描述")
    
    model_config = {"json_schema_extra": {
        "examples": [{
            "target_item": "person"
        }]
    }}


class DetectedObjectResponse(BaseModel):
    """偵測到的物件回應"""
    box_2d: List[int] = Field(..., description="2D 邊界框座標 [y_min, x_min, y_max, x_max]")
    label: str = Field(..., description="物件標籤")


class ImageAnnotationResponse(BaseModel):
    """圖片標註回應"""
    status: str = Field(default="success", description="狀態")
    message: str = Field(..., description="訊息")
    total_detected: int = Field(..., description="偵測到的物件總數")
    objects: List[DetectedObjectResponse] = Field(..., description="偵測到的物件列表")
    annotated_images: List[str] = Field(default_factory=list, description="標註圖片檔案路徑")
