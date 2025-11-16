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
from app.services.annotating_service import ImageAnnotationService
from app.models import (
    QuestionRequest,
    QuestionResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse,
    ImageAnnotationRequest,
    ImageAnnotationResponse,
    DetectedObjectResponse,
    CollectionsResponse,
    CollectionInfo,
    SwitchCollectionRequest,
    SwitchCollectionResponse
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
annotation_service: ImageAnnotationService = None


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


@app.get("/api/collections", response_model=CollectionsResponse)
async def get_collections():
    """
    獲取所有可用的 collections
    """
    try:
        if not db_service:
            raise HTTPException(status_code=503, detail="資料庫服務未初始化")
        
        collections = db_service.list_collections()
        current_collection = db_service.get_current_collection_name()
        
        collection_info = [
            CollectionInfo(name=col["name"], count=col["count"])
            for col in collections
        ]
        
        logger.info(f"返回 {len(collection_info)} 個 collections")
        
        return CollectionsResponse(
            status="success",
            current_collection=current_collection,
            collections=collection_info
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取 collections 時發生錯誤：{e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"獲取 collections 時發生錯誤：{str(e)}"
        )


@app.post("/api/switch-collection", response_model=SwitchCollectionResponse)
async def switch_collection(request: SwitchCollectionRequest):
    """
    切換到不同的 collection
    """
    try:
        if not db_service:
            raise HTTPException(status_code=503, detail="資料庫服務未初始化")
        
        collection_name = request.collection_name.strip()
        
        if not collection_name:
            raise HTTPException(status_code=400, detail="Collection 名稱不能為空")
        
        logger.info(f"正在切換到 collection：{collection_name}")
        
        # 切換 collection
        db_service.switch_collection(collection_name)
        
        # 獲取文件數量
        doc_count = db_service.count()
        
        logger.info(f"成功切換到 collection：{collection_name}，文件數量：{doc_count}")
        
        return SwitchCollectionResponse(
            status="success",
            message=f"已切換到 {collection_name}",
            current_collection=collection_name,
            document_count=doc_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切換 collection 時發生錯誤：{e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"切換 collection 時發生錯誤：{str(e)}"
        )


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


# ========== 圖片上傳與標註 API ==========

from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil


@app.post("/api/upload")
async def upload_image_file(file: UploadFile = File(...)):
    """
    上傳圖片檔案（不進行標註）
    """
    try:
        # 檢查檔案類型
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="不支援的檔案類型，僅支援 JPG、PNG、GIF、WEBP"
            )
        
        # 建立上傳目錄
        upload_dir = settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 儲存檔案
        file_path = upload_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size
        
        logger.info(f"檔案已上傳：{file.filename}，大小：{file_size} bytes")
        
        return {
            "status": "success",
            "message": "圖片上傳成功",
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上傳圖片時發生錯誤：{e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"上傳圖片時發生錯誤：{str(e)}"
        )


@app.post("/api/annotate-image", response_model=ImageAnnotationResponse)
async def annotate_image(
    file: UploadFile = File(...),
    target_item: str = Form(default="objects")
):
    """
    上傳並標註圖片
    """
    try:
        # 檢查 Google API Key
        if not settings.google_api_key:
            raise HTTPException(
                status_code=400,
                detail="Google API Key 未設定，請先在 .env 檔案中設定 GOOGLE_API_KEY"
            )
        
        # 檢查檔案類型
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="不支援的檔案類型，僅支援 JPG、PNG、GIF、WEBP"
            )
        
        # 建立上傳目錄
        upload_dir = settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 儲存上傳的檔案
        file_path = upload_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"圖片已上傳：{file.filename}")
        
        # 初始化標註服務
        global annotation_service
        if not annotation_service:
            annotation_service = ImageAnnotationService()
        
        # 執行物件偵測
        detected_objects = annotation_service.detect_objects(
            image_path=str(file_path),
            target_item=target_item
        )
        
        # 檢查是否偵測到物件
        if not detected_objects:
            logger.warning(f"圖片中未偵測到指定的物件：{target_item}")
            return ImageAnnotationResponse(
                status="success",
                message=f"圖片分析完成，但未在圖片中找到 '{target_item}' 相關的物件",
                total_detected=0,
                objects=[],
                annotated_images=[]
            )
        
        # 儲存標註圖片
        annotated_files = annotation_service.annotate_image(
            image_path=str(file_path),
            target_item=target_item
        )
        
        # 轉換為回應格式
        objects_response = [
            DetectedObjectResponse(
                box_2d=obj.box_2d,
                label=obj.label
            )
            for obj in detected_objects
        ]
        
        logger.info(f"圖片標註完成，偵測到 {len(detected_objects)} 個物件")
        
        return ImageAnnotationResponse(
            status="success",
            message=f"成功偵測並標註 {len(detected_objects)} 個物件",
            total_detected=len(detected_objects),
            objects=objects_response,
            annotated_images=annotated_files
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"標註圖片時發生錯誤：{e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"標註圖片時發生錯誤：{str(e)}"
        )


@app.get("/api/annotated-images/{filename}")
async def get_annotated_image(filename: str):
    """
    取得標註後的圖片檔案
    """
    try:
        from urllib.parse import quote
        
        output_dir = settings.output_dir / "Annotated_Image"
        file_path = output_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="找不到圖片檔案")
        
        # 使用 RFC 5987 編碼來支援中文檔名
        encoded_filename = quote(filename)
        
        return FileResponse(
            path=str(file_path),
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"讀取標註圖片時發生錯誤：{e}")
        raise HTTPException(
            status_code=500,
            detail=f"讀取標註圖片時發生錯誤：{str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
