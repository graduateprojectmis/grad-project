"""
向量嵌入服務
支援 OpenAI 和 Google Gemini
"""

from typing import List, Union, Generator, Optional, Callable
from abc import ABC, abstractmethod
import time
from openai import OpenAI

import google.generativeai as genai

from app.core.logger import get_logger
from app.core.exceptions import EmbeddingError, APIKeyError
from app.config import get_settings

logger = get_logger(__name__)

# 預設批次大小
DEFAULT_BATCH_SIZE_OPENAI = 100  # OpenAI 建議單次最多處理 100 個
DEFAULT_BATCH_SIZE_GEMINI = 50   # Gemini 較保守的批次大小

# API 限制
MAX_BATCH_SIZE_OPENAI = 2048  # OpenAI 單次請求最大數量
MAX_BATCH_SIZE_GEMINI = 100   # Gemini 單次請求最大數量


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

    @abstractmethod
    def generate_embedding_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        delay_between_batches: float = 0.1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """
        批次生成嵌入向量

        Args:
            texts: 文字列表
            batch_size: 每批處理的數量
            delay_between_batches: 批次間的延遲時間（秒）
            progress_callback: 進度回調函數，接收 (已完成數量, 總數量)

        Returns:
            嵌入向量列表
        """
        pass

    def _chunk_list(self, lst: List, chunk_size: int) -> Generator[List, None, None]:
        """將列表分割成多個批次"""
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]


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
            
        Note:
            如果輸入超過 API 限制 (2048 個)，會自動分批處理
        """
        try:
            # 確保輸入是列表
            if isinstance(text, str):
                text = [text]

            # 檢查是否超過 API 限制，若超過則自動分批
            if len(text) > MAX_BATCH_SIZE_OPENAI:
                logger.warning(
                    f"輸入數量 ({len(text)}) 超過 API 限制 ({MAX_BATCH_SIZE_OPENAI})，"
                    f"將自動分批處理"
                )
                return self.generate_embedding_batch(
                    text, batch_size=DEFAULT_BATCH_SIZE_OPENAI
                )

            logger.debug(f"正在生成 {len(text)} 個文字的嵌入向量")

            response = self.client.embeddings.create(model=self.model, input=text)

            embeddings = [item.embedding for item in response.data]
            logger.debug(f"成功生成 {len(embeddings)} 個嵌入向量")

            return embeddings

        except Exception as e:
            logger.error(f"生成 OpenAI 嵌入向量時發生錯誤：{e}")
            raise EmbeddingError(f"生成嵌入向量失敗：{str(e)}")

    def generate_embedding_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        delay_between_batches: float = 0.1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """
        批次生成 OpenAI 嵌入向量

        Args:
            texts: 文字列表
            batch_size: 每批處理的數量，預設為 100
            delay_between_batches: 批次間的延遲時間（秒），預設 0.1 秒
            progress_callback: 進度回調函數，接收 (已完成數量, 總數量)

        Returns:
            嵌入向量列表
        """
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE_OPENAI

        if not texts:
            return []

        total = len(texts)
        all_embeddings = []
        processed = 0

        logger.info(f"開始批次處理 {total} 個文字，每批 {batch_size} 個")

        try:
            for batch in self._chunk_list(texts, batch_size):
                batch_embeddings = self.generate_embedding(batch)
                all_embeddings.extend(batch_embeddings)
                processed += len(batch)

                if progress_callback:
                    progress_callback(processed, total)

                logger.debug(f"批次進度：{processed}/{total}")

                # 如果還有下一批，則延遲
                if processed < total and delay_between_batches > 0:
                    time.sleep(delay_between_batches)

            logger.info(f"批次處理完成，共生成 {len(all_embeddings)} 個嵌入向量")
            return all_embeddings

        except Exception as e:
            logger.error(f"批次生成 OpenAI 嵌入向量時發生錯誤：{e}")
            raise EmbeddingError(f"批次生成嵌入向量失敗：{str(e)}")


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
            
        Note:
            如果輸入超過 API 限制 (100 個)，會自動分批處理
        """
        try:
            # 確保輸入是列表
            if isinstance(text, str):
                text = [text]

            # 檢查是否超過 API 限制，若超過則自動分批
            if len(text) > MAX_BATCH_SIZE_GEMINI:
                logger.warning(
                    f"輸入數量 ({len(text)}) 超過 API 限制 ({MAX_BATCH_SIZE_GEMINI})，"
                    f"將自動分批處理"
                )
                return self.generate_embedding_batch(
                    text, batch_size=DEFAULT_BATCH_SIZE_GEMINI
                )

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

    def generate_embedding_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        delay_between_batches: float = 0.2,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """
        批次生成 Gemini 嵌入向量

        Args:
            texts: 文字列表
            batch_size: 每批處理的數量，預設為 50
            delay_between_batches: 批次間的延遲時間（秒），預設 0.2 秒
            progress_callback: 進度回調函數，接收 (已完成數量, 總數量)

        Returns:
            嵌入向量列表
        """
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE_GEMINI

        if not texts:
            return []

        total = len(texts)
        all_embeddings = []
        processed = 0

        logger.info(f"開始批次處理 {total} 個文字，每批 {batch_size} 個")

        try:
            for batch in self._chunk_list(texts, batch_size):
                batch_embeddings = self.generate_embedding(batch)
                all_embeddings.extend(batch_embeddings)
                processed += len(batch)

                if progress_callback:
                    progress_callback(processed, total)

                logger.debug(f"批次進度：{processed}/{total}")

                # 如果還有下一批，則延遲（Gemini API 可能有較嚴格的速率限制）
                if processed < total and delay_between_batches > 0:
                    time.sleep(delay_between_batches)

            logger.info(f"批次處理完成，共生成 {len(all_embeddings)} 個嵌入向量")
            return all_embeddings

        except Exception as e:
            logger.error(f"批次生成 Gemini 嵌入向量時發生錯誤：{e}")
            raise EmbeddingError(f"批次生成嵌入向量失敗：{str(e)}")


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

    def generate_embedding_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        delay_between_batches: float = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """
        批次生成嵌入向量

        適合處理大量文字，會自動分批處理並控制 API 請求頻率。

        Args:
            texts: 文字列表
            batch_size: 每批處理的數量，若為 None 則使用各服務的預設值
                       (OpenAI: 100, Gemini: 50)
            delay_between_batches: 批次間的延遲時間（秒），若為 None 則使用各服務的預設值
                                  (OpenAI: 0.1s, Gemini: 0.2s)
            progress_callback: 進度回調函數，接收 (已完成數量, 總數量)
                              可用於更新進度條或日誌

        Returns:
            嵌入向量列表

        Example:
            >>> service = EmbeddingService(provider="openai")
            >>> texts = ["文字1", "文字2", "文字3", ...]
            >>> 
            >>> # 基本使用
            >>> embeddings = service.generate_embedding_batch(texts)
            >>> 
            >>> # 自訂批次大小和延遲
            >>> embeddings = service.generate_embedding_batch(
            ...     texts,
            ...     batch_size=50,
            ...     delay_between_batches=0.5
            ... )
            >>> 
            >>> # 使用進度回調
            >>> def on_progress(done, total):
            ...     print(f"進度: {done}/{total} ({done/total*100:.1f}%)")
            >>> embeddings = service.generate_embedding_batch(
            ...     texts,
            ...     progress_callback=on_progress
            ... )
        """
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if delay_between_batches is not None:
            kwargs["delay_between_batches"] = delay_between_batches
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback

        return self.service.generate_embedding_batch(texts, **kwargs)
