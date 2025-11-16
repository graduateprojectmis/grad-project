"""
資料庫服務
管理 ChromaDB 操作
"""
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.api.models.Collection import Collection

from app.core.logger import get_logger
from app.core.exceptions import DatabaseError
from app.config import get_settings

logger = get_logger(__name__)


class DatabaseService:
    """ChromaDB 資料庫服務"""
    
    def __init__(self, db_path: str = None, collection_name: str = None):
        """
        初始化資料庫服務
        
        Args:
            db_path: 資料庫路徑
            collection_name: 集合名稱
        """
        settings = get_settings()
        self.db_path = db_path or settings.chroma_db_path
        self.collection_name = collection_name or settings.chroma_collection_name
        
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[Collection] = None
        
        self._initialize()
    
    def _initialize(self):
        """初始化資料庫連接"""
        try:
            logger.info(f"正在初始化 ChromaDB，路徑：{self.db_path}")
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            logger.info(
                f"ChromaDB 初始化成功，集合：{self.collection_name}，"
                f"文件數量：{self.collection.count()}"
            )
        except Exception as e:
            logger.error(f"初始化 ChromaDB 時發生錯誤：{e}")
            raise DatabaseError(f"資料庫初始化失敗：{str(e)}")
    
    def insert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """
        插入資料到資料庫
        
        Args:
            ids: 文件 ID 列表
            documents: 文件內容列表
            embeddings: 嵌入向量列表
            metadatas: 元資料列表
        """
        try:
            if metadatas is None:
                metadatas = [{}] * len(ids)
            
            logger.info(f"正在插入 {len(ids)} 筆資料到 ChromaDB")
            
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"成功插入 {len(ids)} 筆資料")
            
        except Exception as e:
            logger.error(f"插入資料時發生錯誤：{e}")
            raise DatabaseError(f"插入資料失敗：{str(e)}")
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 1
    ) -> Tuple[List[str], List[float]]:
        """
        查詢相似文件
        
        Args:
            query_embedding: 查詢嵌入向量
            n_results: 返回結果數量
            
        Returns:
            (文件列表, 距離列表)
        """
        try:
            logger.debug(f"正在查詢相似文件，返回 {n_results} 個結果")
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # 提取結果
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            logger.debug(f"查詢完成，找到 {len(documents)} 個結果")
            
            return documents, distances
            
        except Exception as e:
            logger.error(f"查詢資料時發生錯誤：{e}")
            raise DatabaseError(f"查詢失敗：{str(e)}")
    
    def count(self) -> int:
        """
        獲取資料庫文件數量
        
        Returns:
            文件數量
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"獲取文件數量時發生錯誤：{e}")
            return 0
    
    def clear(self) -> None:
        """清空集合"""
        try:
            logger.warning(f"正在清空集合：{self.collection_name}")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name
            )
            logger.info("集合已清空")
        except Exception as e:
            logger.error(f"清空集合時發生錯誤：{e}")
            raise DatabaseError(f"清空集合失敗：{str(e)}")
    
    def switch_collection(self, collection_name: str) -> None:
        """
        切換到不同的集合
        
        Args:
            collection_name: 集合名稱
        """
        try:
            logger.info(f"正在切換到集合：{collection_name}")
            self.collection_name = collection_name
            self.collection = self.client.get_or_create_collection(
                name=collection_name
            )
            logger.info(
                f"成功切換到集合：{collection_name}，"
                f"文件數量：{self.collection.count()}"
            )
        except Exception as e:
            logger.error(f"切換集合時發生錯誤：{e}")
            raise DatabaseError(f"切換集合失敗：{str(e)}")
    
    def list_collections(self) -> List[Dict[str, any]]:
        """
        列出所有可用的集合
        
        Returns:
            集合資訊列表 [{"name": "collection_name", "count": 123}, ...]
        """
        try:
            collections = self.client.list_collections()
            result = []
            for col in collections:
                result.append({
                    "name": col.name,
                    "count": col.count()
                })
            logger.info(f"找到 {len(result)} 個集合")
            return result
        except Exception as e:
            logger.error(f"列出集合時發生錯誤：{e}")
            raise DatabaseError(f"列出集合失敗：{str(e)}")
    
    def get_current_collection_name(self) -> str:
        """
        獲取當前集合名稱
        
        Returns:
            集合名稱
        """
        return self.collection_name
    
    def close(self) -> None:
        """關閉資料庫連接"""
        logger.info("關閉 ChromaDB 連接")
        self.client = None
        self.collection = None
