import os
import sys
import openai

# 動態添加專案根目錄到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ChromaDB import initialize_chroma_db, query_chromadb

# 設定 OpenAI API Key (使用 openai==0.28.0 版本)
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_with_context(question: str, top_k: int = 1):
    """
    使用 ChromaDB 查詢並回答問題。
    """
    # 初始化 ChromaDB
    db_path = "./chroma_db"
    collection_name = "text_embedding_openai"
    collection = initialize_chroma_db(db_path, collection_name)

    # 查詢 ChromaDB
    results = query_chromadb(collection, question, n_results=top_k)

    prompt = f"""
    你是一個智慧助理，根據以下文件內容回答問題。
    如果文件中沒有相關資訊，就回答「文件中沒有提到」。

    文件內容：
    {results}

    使用者問題：
    {question}

    請以清楚、自然且簡短的中文回答：
    """

    # 檢查 API Key
    if not openai.api_key:
        raise ValueError("❌ OPENAI_API_KEY 未設定！請先設定 API Key。")
    
    # 呼叫 LLM 生成回覆 (openai==0.28.0 版本語法)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    answer = response.choices[0].message.content.strip()
    return answer

if __name__ == "__main__":
    query = input("請輸入你的問題：")
    response = ask_with_context(query, top_k=1)
    print(f"回答：{response}")