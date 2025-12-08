"""
LLM 服務
使用大型語言模型生成回答
"""

from typing import Optional, List
from openai import OpenAI


from app.core.logger import get_logger
from app.core.exceptions import APIKeyError
from app.config import get_settings

logger = get_logger(__name__)


class LLMService:
    """大型語言模型服務"""

    def __init__(self, api_key: str = None, model: str = None):
        """
        初始化 LLM 服務

        Args:
            api_key: OpenAI API Key
            model: 模型名稱
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.temperature = settings.openai_temperature

        if not self.api_key:
            raise APIKeyError("OpenAI API Key 未設定")

        self.client = OpenAI(api_key=self.api_key)

        logger.info(f"LLM 服務已初始化，使用模型：{self.model}")

    def generate_answer(
        self, question: str, context: str, temperature: Optional[float] = None
    ) -> str:
        """
        根據問題和上下文生成答案

        Args:
            question: 使用者問題
            context: 參考上下文
            temperature: 溫度參數

        Returns:
            生成的答案
        """
        try:
            temp = temperature if temperature is not None else self.temperature

            prompt = self._build_prompt(question, context)

            logger.debug(f"正在生成答案，問題：{question}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
            )

            answer = response.choices[0].message.content.strip()

            logger.debug(f"答案生成完成，長度：{len(answer)} 字元")

            return answer

        except Exception as e:
            logger.error(f"生成答案時發生錯誤：{e}")
            raise Exception(f"生成答案失敗：{str(e)}")

    def _build_prompt(self, question: str, context: str) -> str:
        """
        建立提示詞

        Args:
            question: 使用者問題
            context: 參考上下文

        Returns:
            提示詞
        """
        return f"""你是一個智慧助理，根據以下文件內容回答問題。
如果文件中沒有相關資訊，就回答「文件中沒有提到」。

文件內容：
{context}

使用者問題：
{question}

請以清楚、自然且簡短的中文回答："""

    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """
        生成文字摘要

        Args:
            text: 原始文字
            max_length: 最大長度

        Returns:
            摘要文字
        """
        try:
            prompt = f"請將以下內容摘要為不超過 {max_length} 字的中文：\n\n{text}"

            logger.debug("正在生成摘要")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            summary = response.choices[0].message.content.strip()

            logger.debug("摘要生成完成")

            return summary

        except Exception as e:
            logger.error(f"生成摘要時發生錯誤：{e}")
            raise Exception(f"生成摘要失敗：{str(e)}")


class QueryDecompositionService:
    """問題拆解服務 - 使用 Gemini 模型"""

    DECOMPOSITION_PROMPT = """You are an expert Query Decomposition Engine for a RAG (Retrieval-Augmented Generation) system. 

Your task is to analyze the user's input query and break it down into 3 to 5 distinct, high-quality sub-queries to maximize search retrieval accuracy.

# Rules:
1. Break down complex logic into simpler, retrieval-friendly facts.
2. If the query involves comparison, generate separate queries for each entity.
3. Make implicit context explicit (e.g., clarify "it", "latest", or specific versions).
4. Use synonyms or technical terms where appropriate to broaden coverage.
5. Maintain the same language as the user's input query.

# Output Format:
- Return ONLY a raw JSON list of strings.
- Format: ["sub-query 1", "sub-query 2", "sub-query 3"]
- Do NOT use markdown code blocks (no ```json).
- Do NOT include any introductory or concluding text.
- Do NOT output the original query.

Input:
"""

    def __init__(self, api_key: str = None, model: str = None):
        """
        初始化問題拆解服務

        Args:
            api_key: Google API Key
            model: Gemini 模型名稱
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("請安裝 google-generativeai 套件：pip install google-generativeai")

        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        self.model = model or settings.gemini_model
        self.temperature = settings.gemini_temperature

        if not self.api_key:
            raise APIKeyError("Google API Key 未設定")

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

        logger.info(f"問題拆解服務已初始化，使用模型：{self.model}")

    def decompose_query(self, query: str, temperature: Optional[float] = None) -> List[str]:
        """
        將使用者查詢拆解為多個子查詢

        Args:
            query: 使用者原始查詢
            temperature: 溫度參數

        Returns:
            拆解後的子查詢列表
        """
        import json

        try:
            temp = temperature if temperature is not None else self.temperature
            prompt = self.DECOMPOSITION_PROMPT + query

            logger.debug(f"正在拆解查詢：{query}")

            response = self.client.generate_content(
                prompt,
                generation_config={"temperature": temp}
            )

            result_text = response.text.strip()
            
            # 解析 JSON 結果
            sub_queries = json.loads(result_text)

            if not isinstance(sub_queries, list):
                raise ValueError("回應格式錯誤，預期為列表")

            logger.debug(f"查詢拆解完成，產生 {len(sub_queries)} 個子查詢")
            logger.info(f"子查詢列表：{sub_queries}")

            return sub_queries

        except json.JSONDecodeError as e:
            logger.error(f"解析 JSON 時發生錯誤：{e}，原始回應：{result_text}")
            raise Exception(f"問題拆解失敗：回應格式錯誤")
        except Exception as e:
            logger.error(f"拆解查詢時發生錯誤：{e}")
            raise Exception(f"問題拆解失敗：{str(e)}")
