"""
向量嵌入服務
支援 OpenAI 和 Google Gemini
"""

from typing import List, Union
from abc import ABC, abstractmethod
from openai import OpenAI

import google.generativeai as genai

from app.core.logger import get_logger
from app.core.exceptions import EmbeddingError, APIKeyError
from app.config import get_settings

logger = get_logger(__name__)


class BaseEmbeddingService(ABC):
    """嵌入服務基礎類別"""

    @abstractmethod
    def generate_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        生成嵌入向量

        Args:
            text: 單一文字或文字列表

        Returns:
            嵌入向量列表
        """
        pass


class OpenAIEmbeddingService(BaseEmbeddingService):
    """OpenAI 嵌入服務"""

    def __init__(self, api_key: str = None, model: str = None):
        """
        初始化 OpenAI 嵌入服務

        Args:
            api_key: OpenAI API Key
            model: 模型名稱
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model

        if not self.api_key:
            raise APIKeyError("OpenAI API Key 未設定")

        self.client = OpenAI(api_key=self.api_key)

        logger.info(f"OpenAI 嵌入服務已初始化，使用模型：{self.model}")

    def generate_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        生成 OpenAI 嵌入向量

        Args:
            text: 單一文字或文字列表

        Returns:
            嵌入向量列表
        """
        try:
            # 確保輸入是列表
            if isinstance(text, str):
                text = [text]

            logger.debug(f"正在生成 {len(text)} 個文字的嵌入向量")

            response = self.client.embeddings.create(model=self.model, input=text)

            embeddings = [item.embedding for item in response.data]
            logger.debug(f"成功生成 {len(embeddings)} 個嵌入向量")

            return embeddings

        except Exception as e:
            logger.error(f"生成 OpenAI 嵌入向量時發生錯誤：{e}")
            raise EmbeddingError(f"生成嵌入向量失敗：{str(e)}")


class GeminiEmbeddingService(BaseEmbeddingService):
    """Google Gemini 嵌入服務"""

    def __init__(self, api_key: str = None, model: str = "models/embedding-001"):
        """
        初始化 Gemini 嵌入服務

        Args:
            api_key: Google API Key
            model: 模型名稱
        """
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        self.model = model

        if not self.api_key:
            raise APIKeyError("Google API Key 未設定")

        genai.configure(api_key=self.api_key)
        logger.info(f"Gemini 嵌入服務已初始化，使用模型：{self.model}")

    def generate_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        生成 Gemini 嵌入向量

        Args:
            text: 單一文字或文字列表

        Returns:
            嵌入向量列表
        """
        try:
            # 確保輸入是列表
            if isinstance(text, str):
                text = [text]

            logger.debug(f"正在生成 {len(text)} 個文字的嵌入向量")

            embeddings = []
            for t in text:
                result = genai.embed_content(
                    model=self.model, content=t, task_type="retrieval_document"
                )
                embeddings.append(result["embedding"])

            logger.debug(f"成功生成 {len(embeddings)} 個嵌入向量")

            return embeddings

        except Exception as e:
            logger.error(f"生成 Gemini 嵌入向量時發生錯誤：{e}")
            raise EmbeddingError(f"生成嵌入向量失敗：{str(e)}")


class EmbeddingService:
    """統一的嵌入服務介面"""

    def __init__(self, provider: str = "openai", **kwargs):
        """
        初始化嵌入服務

        Args:
            provider: 服務提供者 ('openai' 或 'gemini')
            **kwargs: 傳遞給具體服務的參數
        """
        self.provider = provider.lower()

        if self.provider == "openai":
            self.service = OpenAIEmbeddingService(**kwargs)
        elif self.provider == "gemini":
            self.service = GeminiEmbeddingService(**kwargs)
        else:
            raise ValueError(f"不支援的嵌入服務提供者：{provider}")

        logger.info(f"嵌入服務已初始化，提供者：{self.provider}")

    def generate_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        生成嵌入向量

        Args:
            text: 單一文字或文字列表

        Returns:
            嵌入向量列表
        """
        return self.service.generate_embedding(text)
