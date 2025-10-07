import chromadb
from load_save_data import load_json_data
from generate_embedding_openai import EmbeddingGenerator
import dotenv
import os

dotenv.load_dotenv()

def initialize_chroma_db(db_path, collection_name):
    """
    初始化 ChromaDB 並取得指定的 collection。
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def prepare_data_for_insertion(data):
    """
    準備資料以插入到 ChromaDB 中。
    """
    ids = [f"chunk_{i}" for i in range(len(data))]
    documents = [item["chunk"] for item in data]
    embeddings = [item["embedding"] for item in data]
    return ids, documents, embeddings

def insert_data_into_chromadb(collection, ids, documents, embeddings):
    """
    將資料插入到 ChromaDB 中。
    """
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )
    print(f"✅ 已成功將 {len(ids)} 筆資料存入 ChromaDB！")

# 初始化 EmbeddingGenerator
embedding_generator = EmbeddingGenerator(api_key=os.getenv("OPENAI_API_KEY"))

def query_chromadb(collection, query_text, n_results=1):
    """
    從 ChromaDB 中查詢資料，先將查詢文字相量化，並顯示相似度最高的 chunk。
    """
    # 使用 EmbeddingGenerator 的 generate_embedding 方法
    query_embedding = embedding_generator.generate_embedding(query_text)

    # 如果嵌入向量是嵌套列表，展平它
    if isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]

    # 確保 query_embedding 是 list[float]
    if not isinstance(query_embedding, list) or not all(isinstance(x, (float, int)) for x in query_embedding):
        raise ValueError("查詢嵌入向量的格式不正確，應為 list[float]！")

    # 使用嵌入向量進行查詢
    results = collection.query(
        query_embeddings=[query_embedding],  # 必須是 list[list[float]]
        n_results=n_results
    )

    if results["documents"]:
        top_document = results["documents"][0][0]  # 取得相似度最高的 chunk
        return top_document
    else:
        return "沒有找到相關的內容。"

if __name__ == "__main__":
    input_file = "output/json/text_embedding_openai.json"
    db_path = "./chroma_db"
    collection_name = "text_embedding_openai"
    query_text = "如何使用Airpods 4"  # ← 你想搜尋的內容
    n_results = 1                   # ← 只顯示相似度最高的結果

    # 載入資料
    data = load_json_data(input_file)

    # 初始化 ChromaDB
    collection = initialize_chroma_db(db_path, collection_name)

    # 準備資料並插入到 ChromaDB（如果尚未插入）
    ids, documents, embeddings = prepare_data_for_insertion(data)
    insert_data_into_chromadb(collection, ids, documents, embeddings)

    # 查詢範例
    query_chromadb(collection, query_text, n_results)


