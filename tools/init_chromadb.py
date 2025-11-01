"""
ChromaDB 初始化腳本
自動初始化 ChromaDB 資料庫，不需要用戶輸入
"""
import os
import sys
from pathlib import Path

# 動態添加專案根目錄到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.ChromaDB import initialize_chroma_db, prepare_data_for_insertion, insert_data_into_chromadb

def main():
    """初始化 ChromaDB 資料庫"""
    print("=" * 50)
    print("  ChromaDB 資料庫初始化")
    print("=" * 50)
    print()
    
    # 設定路徑
    input_file = "output/json/text_embedding_openai.json"
    db_path = "./web/backend/chroma_db"  # 注意：資料庫在 backend 目錄
    collection_name = "text_embedding_openai"
    
    # 檢查 JSON 檔案是否存在
    if not Path(input_file).exists():
        print(f"❌ 錯誤：找不到資料檔案 {input_file}")
        print("請先執行資料生成腳本")
        return False
    
    print(f"📂 資料檔案：{input_file}")
    print(f"📂 資料庫路徑：{db_path}")
    print(f"📦 集合名稱：{collection_name}")
    print()
    
    try:
        # 初始化 ChromaDB
        print("🔧 正在初始化 ChromaDB...")
        collection = initialize_chroma_db(db_path, collection_name)
        
        # 檢查資料是否已插入
        current_count = collection.count()
        print(f"📊 目前資料筆數：{current_count}")
        
        if current_count == 0:
            print("📥 正在載入資料...")
            ids, documents, embeddings, metadatas = prepare_data_for_insertion(input_file)
            
            print(f"📝 準備插入 {len(ids)} 筆資料...")
            insert_data_into_chromadb(collection, ids, documents, embeddings, metadatas)
            
            final_count = collection.count()
            print(f"✅ 成功插入 {final_count} 筆資料到 ChromaDB")
        else:
            print("✅ ChromaDB 資料已存在，無需重新插入")
        
        print()
        print("=" * 50)
        print("  初始化完成！")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ 初始化失敗：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

