"""
LLM 服務
使用大型語言模型生成回答
"""
from typing import Optional, List
import openai

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
        
        openai.api_key = self.api_key
        logger.info(f"LLM 服務已初始化，使用模型：{self.model}")
    
    def generate_answer(
        self,
        question: str,
        context: str,
        temperature: Optional[float] = None
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
            
            response = openai.ChatCompletion.create(
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
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 100
    ) -> str:
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
            
            response = openai.ChatCompletion.create(
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
