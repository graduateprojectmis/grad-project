"""
FastAPI 主應用程式
重構後的 API 結構
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from pathlib import Path

from app.config import get_settings
from app.core.logger import setup_logger, get_logger
from app.core.exceptions import AppException
from app.services import DatabaseService, EmbeddingService, LLMService
from app.models import (
    QuestionRequest,
    QuestionResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse
)
from app import __version__

# 初始化設定
settings = get_settings()

# 初始化日誌
setup_logger(
    name="app",
    level=settings.log_level,
    log_file=settings.log_file
)
logger = get_logger(__name__)

# 全域服務實例
db_service: DatabaseService = None
embedding_service: EmbeddingService = None
llm_service: LLMService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    global db_service, embedding_service, llm_service
    
    logger.info("正在啟動應用程式...")
    
    try:
        # 初始化資料庫服務
        db_service = DatabaseService()
        logger.info("資料庫服務已初始化")
        
        # 檢查是否有資料
        if db_service.count() == 0:
            logger.warning("⚠️ ChromaDB 中沒有資料！請先執行資料初始化。")
        
        # 初始化嵌入服務（如果有 API Key）
        if settings.openai_api_key:
            try:
                embedding_service = EmbeddingService(provider="openai")
                logger.info("嵌入服務已初始化")
            except Exception as e:
                logger.warning(f"嵌入服務初始化失敗：{e}")
        
        # 初始化 LLM 服務（如果有 API Key）
        if settings.openai_api_key:
            try:
                llm_service = LLMService()
                logger.info("LLM 服務已初始化")
            except Exception as e:
                logger.warning(f"LLM 服務初始化失敗：{e}")
        
        logger.info("應用程式啟動完成")
        
    except Exception as e:
        logger.error(f"應用程式啟動失敗：{e}")
        raise
    
    yield
    
    # 關閉資源
    logger.info("正在關閉應用程式...")
    if db_service:
        db_service.close()
    logger.info("應用程式已關閉")


# 建立 FastAPI 應用
app = FastAPI(
    title="AirPods Q&A API",
    description="AirPods 智慧問答系統 API（重構版）",
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全域例外處理
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """處理應用程式例外"""
    logger.error(f"應用程式錯誤：{exc.message}", extra=exc.details)
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """處理一般例外"""
    logger.error(f"未預期的錯誤：{str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "伺服器內部錯誤",
            "details": str(exc) if settings.log_level == "DEBUG" else None
        }
    )


# ========== API 路由 ==========

@app.get("/")
async def root():
    """根路徑"""
    return {
        "name": "AirPods Q&A API",
        "version": __version__,
        "status": "running",
        "docs": "/api/docs",
        "endpoints": {
            "health": "GET /api/health",
            "ask": "POST /api/ask",
            "search": "POST /api/search"
        }
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康檢查"""
    try:
        db_count = db_service.count() if db_service else 0
        
        return HealthResponse(
            status="healthy" if db_service else "unhealthy",
            message="API is running",
            chroma_db_count=db_count,
            version=__version__
        )
    except Exception as e:
        logger.error(f"健康檢查失敗：{e}")
        raise HTTPException(status_code=503, detail="服務不可用")


@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    使用 LLM 生成答案
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="問題不能為空")
        
        logger.info(f"收到問題：{request.question}")
        
        # 檢查服務是否已初始化
        if not db_service:
            raise HTTPException(status_code=503, detail="資料庫服務未初始化")
        
        # 動態初始化服務（如果尚未初始化）
        global embedding_service, llm_service
        
        if not embedding_service:
            if not settings.openai_api_key:
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI API Key 未設定，請先設定 API Key"
                )
            embedding_service = EmbeddingService(provider="openai")
        
        if not llm_service:
            if not settings.openai_api_key:
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI API Key 未設定，請先設定 API Key"
                )
            llm_service = LLMService()
        
        # 生成問題嵌入向量
        question_embeddings = embedding_service.generate_embedding([request.question])
        question_embedding = question_embeddings[0]
        
        # 查詢相似文件
        documents, distances = db_service.query(
            query_embedding=question_embedding,
            n_results=request.top_k
        )
        
        if not documents:
            return QuestionResponse(
                question=request.question,
                answer="抱歉，我在資料庫中找不到相關資訊。",
                status="no_results"
            )
        
        # 組合上下文
        context = "\n\n".join(documents)
        
        # 生成答案
        answer = llm_service.generate_answer(
            question=request.question,
            context=context
        )
        
        logger.info(f"答案生成完成")
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            status="success",
            source_chunks=documents
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"處理問題時發生錯誤：{e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"生成答案時發生錯誤：{str(e)}"
        )


@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    搜尋相關文件片段
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="查詢不能為空")
        
        logger.info(f"收到搜尋請求：{request.query}")
        
        if not db_service:
            raise HTTPException(status_code=503, detail="資料庫服務未初始化")
        
        # 動態初始化嵌入服務
        global embedding_service
        if not embedding_service:
            if not settings.openai_api_key:
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI API Key 未設定，請先設定 API Key"
                )
            embedding_service = EmbeddingService(provider="openai")
        
        # 生成查詢嵌入向量
        query_embeddings = embedding_service.generate_embedding([request.query])
        query_embedding = query_embeddings[0]
        
        # 查詢相似文件
        documents, distances = db_service.query(
            query_embedding=query_embedding,
            n_results=request.n_results
        )
        
        logger.info(f"搜尋完成，找到 {len(documents)} 個結果")
        
        return SearchResponse(
            query=request.query,
            results=documents,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜尋時發生錯誤：{e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"搜尋時發生錯誤：{str(e)}"
        )


# ========== Admin API (本地端版本) ==========

from app.models.schemas import APIKeyStatusResponse, SetAPIKeyRequest
from dotenv import set_key, unset_key

ENV_FILE_PATH = (Path(__file__).parent.parent.parent / ".env").as_posix()


@app.get("/api/admin/api-key/status", response_model=APIKeyStatusResponse)
async def get_api_key_status():
    """獲取 API Key 狀態（本地端專用）"""
    value = settings.openai_api_key or ""
    masked = (
        (value[:10] + "...") 
        if value.startswith("sk-") and len(value) > 10 
        else ("" if not value else "已設定")
    )
    
    return APIKeyStatusResponse(
        exists=bool(value),
        masked=masked
    )


@app.post("/api/admin/api-key")
async def set_api_key(payload: SetAPIKeyRequest):
    """設定 API Key（本地端專用）"""
    api_key = payload.api_key.strip()
    if not api_key or not api_key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="API Key 格式不正確")
    
    # 寫入 .env 並更新環境變數
    set_key(ENV_FILE_PATH, "OPENAI_API_KEY", api_key)
    os.environ["OPENAI_API_KEY"] = api_key
    settings.openai_api_key = api_key
    
    # 重新初始化服務
    global embedding_service, llm_service
    embedding_service = None
    llm_service = None
    
    logger.info("API Key 已更新")
    
    return {"status": "success", "message": "API Key 已設定"}


@app.delete("/api/admin/api-key")
async def clear_api_key():
    """清除 API Key（本地端專用）"""
    unset_key(ENV_FILE_PATH, "OPENAI_API_KEY")
    os.environ.pop("OPENAI_API_KEY", None)
    settings.openai_api_key = None
    
    # 清除服務
    global embedding_service, llm_service
    embedding_service = None
    llm_service = None
    
    logger.info("API Key 已清除")
    
    return {"status": "success", "message": "API Key 已清除"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
