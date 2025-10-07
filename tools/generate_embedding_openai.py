"""
1. 將結構化資料中的文字拆分為多個 chunk。
2. 使用 OpenAI API 生成每個 chunk 的嵌入向量。
3. 將 chunk 與 embedding 後的結果合併輸出結果爲 JSON 檔。
4. 輸入： 結構化資料 (list of dict)
5. 輸出： 包含 chunk 與 embedding 的 JSON 檔 (list of dict)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .clean_data import preprocess_text
import os
import json
import openai
import dotenv
from .load_save_data import save_to_json, load_json_data

dotenv.load_dotenv()

class EmbeddingGenerator:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_embedding(self, chunks):
        """
        input: chunks (list[str])
        output: list[embedding vectors]
        """
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=chunks
        )
        return [item["embedding"] for item in response["data"]]

# 初始化 EmbeddingGenerator
embedding_generator = EmbeddingGenerator(api_key=os.getenv("OPENAI_API_KEY"))

def split_text(structured_data, chunk_size=600, chunk_overlap=30):
    # 初始化文本切分器，設定分割規則
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["。", "！", "？", "\n", "，", " "]
    )

    chunk_data = []

    for section in structured_data:
        text = section.get("content", "")
        if not text.strip():
            continue
        # 清理原始文字
        cleaned_text = preprocess_text(text)
        # 切分成 chunk
        chunks = text_splitter.split_text(cleaned_text)
        for chunk in chunks:
            chunk_data.append({
                "chunk": chunk,
                "embedding": embedding_generator.generate_embedding([chunk])[0]
            })

    return chunk_data

if __name__ == "__main__":
    input_file = "output/json/airpods_manual_data.json"
    output_file = "output/json/text_embedding_openai.json"

    structured_data = load_json_data(input_file)
    
    chunk_data = split_text(structured_data)
    
    save_to_json(chunk_data, output_file)
