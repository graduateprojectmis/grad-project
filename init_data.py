"""
資料初始化腳本
從網路抓取資料並建立嵌入向量
"""
import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.config import get_settings
from app.core.logger import setup_logger, get_logger
from app.services import DatabaseService, EmbeddingService
from app.utils import split_text, save_json
from app.services.airpods_manual_fetcher import scrape_airpods_manual

# 初始化設定和日誌
settings = get_settings()
setup_logger(level="INFO", log_file=settings.log_file)
logger = get_logger(__name__)


def main():
    """主函數"""
    logger.info("=" * 60)
    logger.info("開始資料初始化")
    logger.info("=" * 60)
    
    # 1. 抓取資料
    logger.info("步驟 1/4: 正在抓取 AirPods 使用手冊...")
    url = "https://support.apple.com/en-us/guide/airpods/welcome/web"
    raw_data = scrape_airpods_manual(url)
    
    if not raw_data:
        logger.error("抓取資料失敗，程式終止")
        return
    
    logger.info(f"成功抓取 {len(raw_data)} 筆資料")
    
    # 2. 初始化服務
    logger.info("步驟 2/4: 初始化嵌入服務和資料庫...")
    
    try:
        embedding_service = EmbeddingService(provider="openai")
        db_service = DatabaseService()
    except Exception as e:
        logger.error(f"初始化服務失敗：{e}")
        logger.error("請確認已設定 OPENAI_API_KEY 環境變數")
        return
    
    # 3. 處理資料並生成嵌入向量
    logger.info("步驟 3/4: 處理資料並生成嵌入向量...")
    
    texts_to_embed = []
    processed_data = []
    
    for item in raw_data:
        title = item.get('title', '')
        content = item.get('content', '')
        
        # 添加標題
        if title:
            texts_to_embed.append(title)
        
        # 分割內容
        chunks = split_text(content)
        texts_to_embed.extend(chunks)
        
        # 準備資料結構
        processed_data.append({
            "title": title,
            "title_embedding": [],
            "chunks": [{"chunk_text": chunk, "chunk_embedding": []} for chunk in chunks]
        })
    
    logger.info(f"待嵌入文字數量：{len(texts_to_embed)}")
    
    # 批次生成嵌入向量
    logger.info("正在批次生成嵌入向量...")
    try:
        all_embeddings = embedding_service.generate_embedding(texts_to_embed)
    except Exception as e:
        logger.error(f"生成嵌入向量失敗：{e}")
        return
    
    # 映射嵌入向量到資料
    logger.info("正在映射嵌入向量...")
    embedding_idx = 0
    for doc in processed_data:
        # 標題嵌入
        doc['title_embedding'] = all_embeddings[embedding_idx]
        embedding_idx += 1
        
        # 片段嵌入
        for chunk in doc['chunks']:
            chunk['chunk_embedding'] = all_embeddings[embedding_idx]
            embedding_idx += 1
    
    # 儲存處理後的資料
    output_file = settings.output_dir / "processed_data.json"
    save_json(processed_data, str(output_file))
    logger.info(f"處理後的資料已儲存至：{output_file}")
    
    # 4. 插入資料到 ChromaDB
    logger.info("步驟 4/4: 插入資料到 ChromaDB...")
    
    # 清空現有資料
    if db_service.count() > 0:
        logger.warning("資料庫中已有資料，正在清空...")
        db_service.clear()
    
    # 準備插入資料
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    id_counter = 0
    
    for item in processed_data:
        title = item.get("title", "")
        chunks = item.get("chunks", [])
        
        for chunk in chunks:
            ids.append(f"chunk_{id_counter}")
            documents.append(chunk["chunk_text"])
            embeddings.append(chunk["chunk_embedding"])
            metadatas.append({
                "type": "chunk",
                "title": title
            })
            id_counter += 1
    
    # 插入到資料庫
    db_service.insert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    logger.info(f"✅ 成功插入 {len(ids)} 筆資料到 ChromaDB")
    logger.info(f"資料庫路徑：{settings.chroma_db_path}")
    logger.info("=" * 60)
    logger.info("資料初始化完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
