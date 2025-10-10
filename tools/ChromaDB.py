import chromadb
import dotenv
import os
from .load_save_data import load_json_data
from .generate_embedding_openai import EmbeddingGenerator

dotenv.load_dotenv()

def initialize_chroma_db(db_path, collection_name):
    """
    初始化 ChromaDB 並取得指定的 collection。
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def prepare_data_for_insertion(input_file):
    """
    準備資料以插入到 ChromaDB 中（僅插入 chunks 的內容）。
    """
    # 載入 JSON 資料
    data = load_json_data(input_file)
    if not data:
        raise ValueError("輸入的 JSON 檔案為空或格式不正確！")

    # 提取 ids, documents, embeddings, metadatas 
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    id_counter = 0

    for item in data:
        chunks = item.get("chunks", [])

        # 儲存 chunk 的資料
        for chunk in chunks:
            ids.append(f"chunk_{id_counter}")
            documents.append(chunk["chunk_text"])
            embeddings.append(chunk["chunk_embedding"])
            metadatas.append({
                "type": "chunk",
            })
            id_counter += 1

    return ids, documents, embeddings, metadatas

def insert_data_into_chromadb(collection, ids, documents, embeddings, metadatas=None):
    """
    將資料插入到 ChromaDB 中。
    """
    if metadatas is None:
        metadatas = [{}] * len(ids)  # 預設為空字典列表

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"✅ 已成功將 {len(ids)} 筆資料存入 ChromaDB！")

# 初始化 EmbeddingGenerator
embedding_generator = EmbeddingGenerator(api_key=os.getenv("OPENAI_API_KEY"))

def query_chromadb(collection, query_text, n_results=1):
    """
    從 ChromaDB 中查詢資料，先將查詢文字相量化，並顯示相似度最高的 chunk。
    """
    query_embedding = embedding_generator.generate_embedding(query_text)

    # 如果嵌入向量是嵌套列表，展平它
    # 例如：[[0.1, 0.2, 0.3]] → [0.1, 0.2, 0.3]
    if query_embedding and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]

    # 確保 query_embedding 是 list[float]
    if not isinstance(query_embedding, list) or not all(isinstance(x, (float, int)) for x in query_embedding):
        raise ValueError("查詢嵌入向量的格式不正確，應為 list[float]！")

    # 使用嵌入向量進行查詢
    # format:
    # {
    #     "ids": [["id1", "id2", ...]],  # 查詢結果的 ID 列表
    #     "documents": [["doc1", "doc2", ...]],  # 查詢結果的文件內容列表
    #     "embeddings": [[embedding1, embedding2, ...]],  # 查詢結果的嵌入向量列表
    #     "metadatas": [[metadata1, metadata2, ...]],  # 查詢結果的元數據列表
    #     "distances": [[distance1, distance2, ...]]  # 查詢結果的相似度距離列表
    # }
    results = collection.query(
        query_embeddings=[query_embedding],  # 必須是 list[list[float]]
        n_results=n_results
    )

    # 嚴謹檢查查詢結果
    if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
        top_document = results["documents"][0][0]  # 取得相似度最高的 chunk
        return top_document
    else:
        return "沒有找到相關的內容。"

if __name__ == "__main__":
    input_file = "output/json/text_embedding_openai.json"
    db_path = "./chroma_db"
    collection_name = "text_embedding_openai" 
    query_text = input("你想搜尋的內容：")  # ← 你想搜尋的內容
    n_results = 1                   # ← 只顯示相似度最高的結果

    # 初始化 ChromaDB
    collection = initialize_chroma_db(db_path, collection_name)

    # 檢查資料是否已插入
    if collection.count() == 0:
        ids, documents, embeddings, metadatas = prepare_data_for_insertion(input_file)
        insert_data_into_chromadb(collection, ids, documents, embeddings, metadatas)
    else:
        print("資料已存在於 ChromaDB 中，跳過插入。")

    # 查詢範例
    result = query_chromadb(collection, query_text, n_results)
    print(f"查詢結果：{result}")


