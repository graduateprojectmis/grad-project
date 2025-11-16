> 基於 FastAPI + ChromaDB + OpenAI 的現代化智慧問答系統
## 📚 目錄

1. [專案概述](#專案概述)
2. [快速開始](#快速開始)
3. [專案架構](#專案架構)
4. [功能特色](#功能特色)
5. [API 使用](#api-使用)
6. [開發指南](#開發指南)
7. [重構總結](#重構總結)
8. [常用命令](#常用命令)
9. [故障排除](#故障排除)

---

## 專案概述

### 核心功能

- ✅ **智慧問答 (RAG)** - 基於向量資料庫的檢索增強生成
- ✅ **語義搜尋** - ChromaDB 向量相似度搜尋
- ✅ **圖片標註** - Google Gemini AI 物件偵測
- ✅ **多模型支援** - OpenAI 和 Gemini 雙引擎
- ✅ **API 管理** - RESTful API 完整文檔

### 技術棧

**後端**:
- FastAPI - 高效能 Web 框架
- ChromaDB - 向量資料庫
- Pydantic - 數據驗證
- OpenAI API - 語言模型
- Google Gemini - 多模態 AI

**前端**:
- React 18 - 現代化 UI 框架
- Vite - 極速開發工具
- Axios - HTTP 客戶端

---

## 快速開始

### 環境需求

- Python 3.8+
- Node.js 16+ (用於 React 前端)
- OpenAI API Key
- Google API Key (可選，用於圖片標註)

### 一鍵啟動

```bash
# 方式 1: 啟動完整系統（後端 + React 前端）
./start-react.sh

# 方式 2: 只啟動後端
python run_api.py

# 方式 3: 使用自動化腳本
./start-new.sh
```

### 詳細安裝步驟

#### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

#### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env 並設定：
# OPENAI_API_KEY=sk-your-key-here
# GOOGLE_API_KEY=your-google-key-here (可選)
```

#### 3. 初始化資料

```bash
python init_data.py
```

#### 4. 啟動服務

**後端**:
```bash
python run_api.py
# 訪問 http://localhost:8000
# API 文檔 http://localhost:8000/docs
```

**React 前端**:
```bash
cd frontend-react
npm install
npm run dev
# 訪問 http://localhost:3000
```

---

## 專案架構

### 目錄結構

```
Grad-Project/
├── app/                          # 主應用程式
│   ├── config/                   # 配置管理
│   │   └── settings.py          # Pydantic Settings
│   ├── core/                     # 核心功能
│   │   ├── logger.py            # 日誌系統
│   │   └── exceptions.py        # 自定義例外
│   ├── models/                   # 資料模型
│   │   └── schemas.py           # Pydantic 模型
│   ├── services/                 # 業務邏輯層
│   │   ├── embedding_service.py # 向量嵌入
│   │   ├── database_service.py  # ChromaDB
│   │   ├── llm_service.py       # LLM 服務
│   │   └── annotating_service.py # 圖片標註
│   ├── utils/                    # 工具函數
│   │   ├── text_processing.py   # 文字處理
│   │   └── file_operations.py   # 檔案操作
│   └── api/                      # API 層
│       └── main.py              # FastAPI 應用
│
├── frontend-react/               # React 前端
│   ├── src/
│   │   ├── components/          # UI 元件
│   │   ├── services/            # API 服務
│   │   └── App.jsx              # 主應用
│   └── package.json
│
├── data/                         # 資料目錄
│   ├── chroma_db/               # 向量資料庫
│   ├── uploads/                 # 上傳檔案
│   └── output/                  # 輸出檔案
│
├── tests/                        # 測試套件
│   ├── test_api.py              # API 測試
│   ├── test_services.py         # 服務測試
│   └── test_*.py                # 其他測試
│
├── docs/                         # 文檔
│   ├── PROJECT_GUIDE.md         # 本文件
│   └── IMAGE_ANNOTATION_GUIDE.md # 圖片標註
│
├── .env                          # 環境變數
├── requirements.txt              # Python 依賴
├── init_data.py                  # 資料初始化
├── run_api.py                    # API 啟動
└── manage_db.py                  # 資料庫管理
```

### 分層架構

```
┌─────────────────────────────────────┐
│         API Layer (FastAPI)         │  ← REST API 端點
├─────────────────────────────────────┤
│       Service Layer (Services)      │  ← 業務邏輯
│  - EmbeddingService                 │
│  - DatabaseService                  │
│  - LLMService                       │
│  - ImageAnnotationService           │
├─────────────────────────────────────┤
│      Data Layer (ChromaDB)          │  ← 資料持久化
└─────────────────────────────────────┘
```

---

## 功能特色

### 1. 智慧問答 (RAG)

```bash
# API 呼叫
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何配對 AirPods？",
    "top_k": 1
  }'
```

**流程**:
1. 將問題轉換為向量
2. 在 ChromaDB 中搜尋相似文件
3. 組合上下文
4. 使用 LLM 生成答案

### 2. 語義搜尋

```bash
# API 呼叫
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "藍牙配對",
    "n_results": 3
  }'
```

### 3. 圖片標註

```bash
# API 呼叫
curl -X POST http://localhost:8000/api/annotate-image \
  -F "file=@image.jpg" \
  -F "target_item=person"
```

**功能**:
- AI 物件偵測
- 自動繪製邊界框
- 標籤標註
- 結果儲存

### 4. API Key 管理

```bash
# 查詢狀態
curl http://localhost:8000/api/admin/api-key/status

# 設定 Key
curl -X POST http://localhost:8000/api/admin/api-key \
  -H "Content-Type: application/json" \
  -d '{"api_key": "sk-..."}'
```

---

## API 使用

### 端點總覽

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 根路徑 |
| `/api/health` | GET | 健康檢查 |
| `/api/ask` | POST | 智慧問答 |
| `/api/search` | POST | 語義搜尋 |
| `/api/upload` | POST | 上傳圖片 |
| `/api/annotate-image` | POST | 圖片標註 |
| `/api/admin/api-key/status` | GET | API Key 狀態 |
| `/api/admin/api-key` | POST | 設定 API Key |
| `/api/admin/api-key` | DELETE | 刪除 API Key |

### API 文檔

訪問 http://localhost:8000/docs 查看完整的 Swagger 文檔。

---

## 開發指南

### 新增功能

#### 1. 新增服務

```python
# app/services/my_service.py
from app.core.logger import get_logger
from app.config import get_settings

logger = get_logger(__name__)

class MyService:
    def __init__(self):
        self.settings = get_settings()
        logger.info("MyService initialized")
    
    def do_something(self):
        logger.info("Doing something...")
        return "result"
```

#### 2. 新增 API 端點

```python
# app/api/main.py
@app.post("/api/my-endpoint")
async def my_endpoint(request: MyRequest):
    try:
        result = my_service.do_something()
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

#### 3. 新增測試

```python
# tests/test_my_service.py
import pytest
from app.services.my_service import MyService

def test_my_service():
    service = MyService()
    result = service.do_something()
    assert result == "expected"
```

### 執行測試

```bash
# 所有測試
python run_tests.py

# 特定測試
pytest tests/test_api.py -v

# 覆蓋率報告
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

### 資料庫管理

```bash
# 查看狀態
python manage_db.py status

# 查詢資料
python manage_db.py query "關鍵字"

# 清空資料庫
python manage_db.py clear

# 重新初始化
python init_data.py
```
