# api.py - 純 API 後端服務，不提供前端
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import dotenv
from pathlib import Path
import shutil
from datetime import datetime

# 動態添加專案根目錄到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from tools.query_with_llm import ask_with_context
from tools.ChromaDB import initialize_chroma_db, query_chromadb

# 載入專案根目錄的 .env 檔案
env_path = os.path.join(project_root, '.env')
dotenv.load_dotenv(env_path)

# 檢查 API Key 是否已設定
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  警告：OPENAI_API_KEY 未設定！")
    print("請在專案根目錄創建 .env 檔案並加入：")
    print("OPENAI_API_KEY=sk-your-api-key-here")
else:
    print(f"✅ API Key 已載入 (開頭: {os.getenv('OPENAI_API_KEY')[:10]}...)")

# 初始化 FastAPI
app = FastAPI(
    title="AirPods Q&A API",
    description="AirPods 智慧問答系統 API",
    version="1.0.0",
    docs_url="/api/docs",  # API 文檔路徑
    redoc_url="/api/redoc"
)

# CORS 設定 - 允許前端專案訪問
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React/Vue 開發伺服器
        "http://localhost:5173",  # Vite 開發伺服器
        "http://localhost:8080",  # 靜態檔案伺服器
        "http://127.0.0.1:5500",  # VS Code Live Server
        "http://127.0.0.1:8080",
        "http://localhost:5500",  # Live Server 預設端口
        # 生產環境時加入您的網域
        # "https://your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域變數
chroma_collection = None

# 建立圖片上傳目錄
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化 ChromaDB"""
    global chroma_collection
    try:
        db_path = "./chroma_db"
        collection_name = "text_embedding_openai"
        chroma_collection = initialize_chroma_db(db_path, collection_name)
        
        if chroma_collection.count() == 0:
            print("⚠️  警告：ChromaDB 中沒有資料！")
            print("請先執行：python tools/ChromaDB.py")
        else:
            print(f"✅ ChromaDB 已載入 {chroma_collection.count()} 筆資料")
    except Exception as e:
        print(f"❌ 初始化 ChromaDB 錯誤：{e}")

# 資料模型
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 1

class QuestionResponse(BaseModel):
    question: str
    answer: str
    status: str

class SearchRequest(BaseModel):
    query: str
    n_results: int = 1

class SearchResponse(BaseModel):
    query: str
    result: str
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str
    chroma_db_count: int

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    file_path: str
    file_size: int

class ApiKeyRequest(BaseModel):
    api_key: str

class ApiKeyResponse(BaseModel):
    status: str
    message: str
    has_key: bool

class ApiKeyStatusResponse(BaseModel):
    status: str
    has_key: bool
    key_preview: str = None

# API 路由
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康檢查"""
    return HealthResponse(
        status="healthy" if chroma_collection else "unhealthy",
        message="API is running",
        chroma_db_count=chroma_collection.count() if chroma_collection else 0
    )

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    使用 LLM 生成完整答案
    """
    try:
        if not request.question or request.question.strip() == "":
            raise HTTPException(status_code=400, detail="問題不能為空")
        
        answer = ask_with_context(request.question, request.top_k)
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"生成答案時發生錯誤：{str(e)}"
        )

@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    僅搜尋相關文件片段
    """
    try:
        if not chroma_collection:
            raise HTTPException(status_code=503, detail="ChromaDB 尚未初始化")
        
        if not request.query or request.query.strip() == "":
            raise HTTPException(status_code=400, detail="查詢不能為空")
        
        result = query_chromadb(chroma_collection, request.query, request.n_results)
        
        return SearchResponse(
            query=request.query,
            result=result,
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"搜尋時發生錯誤：{str(e)}"
        )

@app.post("/api/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    上傳圖片
    支援格式：jpg, jpeg, png, gif, webp
    """
    try:
        # 檢查檔案類型
        allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支援的檔案格式。支援的格式：{', '.join(allowed_extensions)}"
            )
        
        # 生成唯一檔案名稱（使用時間戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # 儲存檔案
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 取得檔案大小
        file_size = file_path.stat().st_size
        
        return UploadResponse(
            status="success",
            message="圖片上傳成功",
            filename=safe_filename,
            file_path=str(file_path),
            file_size=file_size
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"上傳圖片時發生錯誤：{str(e)}"
        )

@app.post("/api/config/apikey", response_model=ApiKeyResponse)
async def save_api_key(request: ApiKeyRequest):
    """
    儲存 API Key 到 .env 文件
    ⚠️ 注意：此端點應該只在本地環境使用
    """
    try:
        api_key = request.api_key.strip()
        
        # 驗證 API Key 格式
        if not api_key:
            raise HTTPException(status_code=400, detail="API Key 不能為空")
        
        if not api_key.startswith('sk-'):
            raise HTTPException(
                status_code=400, 
                detail="API Key 格式不正確，應該以 'sk-' 開頭"
            )
        
        # .env 文件路徑
        env_path = os.path.join(project_root, '.env')
        
        # 讀取現有的 .env 內容（如果存在）
        env_lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                env_lines = f.readlines()
        
        # 更新或添加 OPENAI_API_KEY
        key_found = False
        for i, line in enumerate(env_lines):
            if line.strip().startswith('OPENAI_API_KEY='):
                env_lines[i] = f'OPENAI_API_KEY={api_key}\n'
                key_found = True
                break
        
        if not key_found:
            env_lines.append(f'\n# OpenAI API Key\nOPENAI_API_KEY={api_key}\n')
        
        # 寫入 .env 文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(env_lines)
        
        # 設定檔案權限為 600（僅擁有者可讀寫）
        try:
            os.chmod(env_path, 0o600)
        except:
            pass  # Windows 可能不支援 chmod
        
        # 重新載入環境變數
        dotenv.load_dotenv(env_path, override=True)
        
        # 更新 openai 的 api_key
        import tools.query_with_llm as query_module
        query_module.openai.api_key = api_key
        
        return ApiKeyResponse(
            status="success",
            message="API Key 已成功儲存到 .env 文件",
            has_key=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"儲存 API Key 時發生錯誤：{str(e)}"
        )

@app.delete("/api/config/apikey", response_model=ApiKeyResponse)
async def delete_api_key():
    """
    從 .env 文件中刪除 API Key
    """
    try:
        env_path = os.path.join(project_root, '.env')
        
        if not os.path.exists(env_path):
            return ApiKeyResponse(
                status="success",
                message="API Key 不存在",
                has_key=False
            )
        
        # 讀取並過濾掉 OPENAI_API_KEY
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = [line for line in lines if not line.strip().startswith('OPENAI_API_KEY=')]
        
        # 寫回文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        # 重新載入環境變數
        dotenv.load_dotenv(env_path, override=True)
        
        return ApiKeyResponse(
            status="success",
            message="API Key 已從 .env 文件中刪除",
            has_key=False
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"刪除 API Key 時發生錯誤：{str(e)}"
        )

@app.get("/api/config/apikey/status", response_model=ApiKeyStatusResponse)
async def get_api_key_status():
    """
    檢查 API Key 是否已設定
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key and api_key.startswith('sk-'):
            return ApiKeyStatusResponse(
                status="success",
                has_key=True,
                key_preview=f"{api_key[:10]}..." if len(api_key) > 10 else "sk-***"
            )
        else:
            return ApiKeyStatusResponse(
                status="success",
                has_key=False
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"檢查 API Key 狀態時發生錯誤：{str(e)}"
        )

@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "AirPods Q&A API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "endpoints": {
            "health": "GET /api/health",
            "ask": "POST /api/ask",
            "search": "POST /api/search",
            "upload": "POST /api/upload",
            "save_api_key": "POST /api/config/apikey",
            "delete_api_key": "DELETE /api/config/apikey",
            "api_key_status": "GET /api/config/apikey/status"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
