import json
import google.generativeai as genai
import os
import time
import dotenv

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. 定義資料處理函式 ---

def load_json_data(file_path: str) -> list:
    """從指定的路徑載入 JSON 檔案。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。")
        return []
    except json.JSONDecodeError:
        print(f"錯誤：檔案 '{file_path}' 的 JSON 格式不正確。")
        return []

def process_and_embed_data(raw_data: list, model_name: str = 'models/text-embedding-004') -> list:
    """
    處理原始資料，對標題和內容分塊進行批次向量化。

    Args:
        raw_data: 從 JSON 檔案讀取的原始資料列表。
        model_name: 要使用的 Embedding 模型名稱。

    Returns:
        一個包含標題、分塊及對應向量的處理後資料列表。
    """
    if not raw_data:
        return []

    texts_to_embed = []
    processed_data = []

    print("步驟 1/4：正在準備文字並進行分塊...")
    # 準備所有要向量化的文字 (標題 + chunks)
    for item in raw_data:
        title = item.get('title', '')
        content = item.get('content', '')

        # 將標題加入待處理列表
        texts_to_embed.append(title)
        
        # 將內容分塊，並過濾掉空字串
        chunks = [chunk.strip() for chunk in content.split('\n') if chunk.strip()]
        
        # 將有效的分塊加入待處理列表
        texts_to_embed.extend(chunks)

        # 建立最終輸出的基本結構
        processed_data.append({
            "title": title,
            "title_embedding": [], # 稍後填入
            "chunks": [{"chunk_text": chunk, "chunk_embedding": []} for chunk in chunks]
        })

    print(f"總共需要向量化的文字數量：{len(texts_to_embed)}")

    print("步驟 2/4：正在呼叫 Gemini API 進行批次向量化...")
    # 進行批次向量化 (Batch Embedding)
    try:
        # Gemini API 一次請求的數量上限為 100，若超過需分批處理
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i+batch_size]
            # 為了避免觸及每分鐘請求次數上限，可以在批次間加入短暫延遲
            if i > 0:
                print(f"處理完畢 {i} 個項目，延遲 1 秒...")
                time.sleep(1)
            
            result = genai.embed_content(model=model_name,
                                         content=batch,
                                         task_type="RETRIEVAL_DOCUMENT")
            all_embeddings.extend(result['embedding'])
            
    except Exception as e:
        print(f"呼叫 API 時發生錯誤：{e}")
        return []

    print("步驟 3/4：正在將向量結果映射回原始資料...")
    # 將向量結果對應回 processed_data
    embedding_idx = 0
    for doc in processed_data:
        # 對應 title 的 embedding
        doc['title_embedding'] = all_embeddings[embedding_idx]
        embedding_idx += 1
        
        # 對應每個 chunk 的 embedding
        for chunk in doc['chunks']:
            chunk['chunk_embedding'] = all_embeddings[embedding_idx]
            embedding_idx += 1
            
    print("步驟 4/4：資料處理完成。")
    return processed_data

def save_to_json(data: list, output_file_path: str):
    """將處理好的資料儲存為 JSON 檔案。"""
    if not data:
        print("沒有資料可以儲存。")
        return
        
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"成功將 Embeddings 儲存至 '{output_file_path}'")
    except IOError as e:
        print(f"寫入檔案時發生錯誤：{e}")

# --- 3. 主程式執行流程 ---
if __name__ == "__main__":
    INPUT_FILE = "output/json/airpods_manual_data.json"
    OUTPUT_FILE = "output/json/text_embedding_gemini.json"

    # 載入原始資料
    airpods_manual_data = load_json_data(INPUT_FILE)

    if airpods_manual_data:
        # 處理資料並產生 Embeddings
        final_data_with_embeddings = process_and_embed_data(airpods_manual_data)
        
        # 儲存結果
        save_to_json(final_data_with_embeddings, OUTPUT_FILE)