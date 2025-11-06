# AirPods Q&A 智慧問答系統（重構版 v2.0）

> 🎉 全新架構！基於 FastAPI + ChromaDB + OpenAI 的現代化智慧問答系統

## ✨ 重構亮點

### 🏗️ 專業架構設計
- **分層架構**：清晰的 Config / Core / Services / Models / Utils 分層
- **依賴注入**：使用 FastAPI 的依賴注入系統
- **統一配置**：基於 Pydantic Settings 的環境變數管理
- **完整日誌**：結構化日誌記錄，支援檔案和控制台輸出
- **錯誤處理**：自定義例外類別和全域錯誤處理

### 🚀 核心功能
- ✅ 智慧問答（RAG - Retrieval-Augmented Generation）
- ✅ 語義搜尋（Semantic Search）
- ✅ 向量資料庫（ChromaDB）
- ✅ 多模型支援（OpenAI、Gemini）
- ✅ RESTful API（FastAPI）
- ✅ API Key 管理
- ✅ 健康檢查

## 📁 專案結構

```
Grad-Project/
├── app/                          # 主應用程式
│   ├── __init__.py
│   ├── config/                   # 配置模組
│   │   ├── __init__.py
│   │   └── settings.py          # 環境變數設定
│   ├── core/                     # 核心功能
│   │   ├── __init__.py
│   │   ├── logger.py            # 日誌系統
│   │   └── exceptions.py        # 自定義例外
│   ├── models/                   # 資料模型
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic 模型
│   ├── services/                 # 業務邏輯層
│   │   ├── __init__.py
│   │   ├── embedding_service.py # 嵌入向量服務
│   │   ├── database_service.py  # 資料庫服務
│   │   └── llm_service.py       # LLM 服務
│   ├── utils/                    # 工具函數
│   │   ├── __init__.py
│   │   ├── text_processing.py   # 文字處理
│   │   └── file_operations.py   # 檔案操作
│   └── api/                      # API 層
│       ├── __init__.py
│       └── main.py              # FastAPI 應用
│
├── src/                          # 舊版程式碼（向後相容）
│   ├── tools/                   # 工具腳本
│   └── web/                     # 前端
│       ├── backend/
│       └── frontend/
│
├── data/                         # 資料目錄
│   ├── chroma_db/               # ChromaDB 資料庫
│   ├── output/                  # 輸出檔案
│   └── uploads/                 # 上傳檔案
│
├── logs/                         # 日誌檔案
│   └── app.log
│
├── .env                          # 環境變數（需自行建立）
├── .env.example                  # 環境變數範例
├── requirements-new.txt          # 新版依賴（精簡版）
├── init_data.py                  # 資料初始化腳本
├── run_api.py                    # API 啟動腳本
└── README.md                     # 本文件
```

## 🚀 快速開始

### 1. 環境需求

- Python 3.8+
- OpenAI API Key

### 2. 安裝依賴

```bash
# 使用新版精簡依賴（推薦）
pip install -r requirements-new.txt

# 或使用完整依賴（包含所有舊版功能）
pip install -r requirements.txt
```

### 3. 設定環境變數

```bash
# 複製環境變數範例
cp .env.example .env

# 編輯 .env 並填入您的 API Key
# OPENAI_API_KEY=sk-your-api-key-here
```

### 4. 初始化資料

```bash
# 抓取資料並建立向量資料庫
python init_data.py
```

### 5. 啟動服務

```bash
# 啟動 API 伺服器
python run_api.py
```

服務會在以下位置啟動：
- **API 服務**: http://localhost:8000
- **API 文檔**: http://localhost:8000/api/docs
- **前端介面**: http://localhost:8080（需另外啟動）

### 6. 啟動前端（選用）

```bash
cd src/web/frontend
python -m http.server 8080
```

## 📚 API 使用說明

### 健康檢查

```bash
curl http://localhost:8000/api/health
```

### 智慧問答

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何配對 AirPods？",
    "top_k": 1
  }'
```

### 語義搜尋

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "配對藍牙",
    "n_results": 3
  }'
```

### 管理 API Key

```bash
# 查詢 API Key 狀態
curl http://localhost:8000/api/admin/api-key/status

# 設定 API Key
curl -X POST http://localhost:8000/api/admin/api-key \
  -H "Content-Type: application/json" \
  -d '{"api_key": "sk-..."}'

# 清除 API Key
curl -X DELETE http://localhost:8000/api/admin/api-key
```

## 🔧 配置說明

所有配置都可以通過環境變數設定（`.env` 檔案）：

| 環境變數 | 說明 | 預設值 |
|---------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `GOOGLE_API_KEY` | Google API Key (選用) | - |
| `ADMIN_TOKEN` | 管理 API Token (選用) | - |
| `API_HOST` | API 服務主機 | 0.0.0.0 |
| `API_PORT` | API 服務端口 | 8000 |
| `LOG_LEVEL` | 日誌級別 | INFO |
| `CHROMA_DB_PATH` | ChromaDB 路徑 | ./data/chroma_db |
| `OPENAI_MODEL` | OpenAI 模型 | gpt-4o-mini |
| `CHUNK_SIZE` | 文字片段大小 | 600 |

## 🏗️ 架構說明

### 分層架構

```
┌─────────────────────────────────────┐
│         API Layer (FastAPI)         │  ← REST API 端點
├─────────────────────────────────────┤
│       Service Layer (Services)      │  ← 業務邏輯
│  - EmbeddingService                 │
│  - DatabaseService                  │
│  - LLMService                       │
├─────────────────────────────────────┤
│      Data Layer (ChromaDB)          │  ← 資料持久化
└─────────────────────────────────────┘
```

### 核心服務

1. **EmbeddingService** - 向量嵌入服務
   - 支援 OpenAI 和 Gemini
   - 統一的介面設計
   - 批次處理優化

2. **DatabaseService** - 資料庫服務
   - ChromaDB 操作封裝
   - 插入、查詢、管理
   - 錯誤處理

3. **LLMService** - 語言模型服務
   - 答案生成
   - 文字摘要
   - 提示詞管理

### 配置管理

使用 Pydantic Settings 實現：
- 環境變數自動載入
- 型別驗證
- 預設值管理
- 單例模式

### 日誌系統

結構化日誌記錄：
- 檔案和控制台雙輸出
- 日誌級別控制
- 彩色輸出（控制台）
- 自動日誌輪替

## 🆚 新舊版本對比

| 功能 | 舊版 | 新版 (v2.0) |
|------|------|-------------|
| 架構 | 腳本式 | 分層架構 |
| 配置管理 | 散落各處 | 統一配置（Pydantic） |
| 錯誤處理 | 基本 try-catch | 自定義例外 + 全域處理 |
| 日誌 | print 語句 | 結構化日誌系統 |
| API 文檔 | 無 | 自動生成（Swagger） |
| 依賴注入 | 無 | FastAPI DI |
| 型別提示 | 部分 | 完整型別提示 |
| 測試友好度 | 低 | 高（可模擬服務） |
| 可維護性 | 中 | 高 |

## 🔄 遷移指南

### 從舊版遷移

1. **安裝新版依賴**
   ```bash
   pip install -r requirements-new.txt
   ```

2. **設定環境變數**
   ```bash
   cp .env.example .env
   # 編輯 .env
   ```

3. **遷移資料**
   ```bash
   python init_data.py
   ```

4. **啟動新版 API**
   ```bash
   python run_api.py
   ```

5. **更新前端 API URL**（如需使用前端）
   - 前端仍可使用，只需確保 API URL 正確

## 📊 效能優化

- ✅ 批次嵌入向量生成（減少 API 呼叫）
- ✅ 資料庫連接池
- ✅ 快取機制（Pydantic Settings）
- ✅ 非同步處理（FastAPI）

## 🛠️ 開發指南

### 新增功能

1. **新增服務**
   ```python
   # app/services/my_service.py
   from app.core.logger import get_logger
   
   logger = get_logger(__name__)
   
   class MyService:
       def do_something(self):
           logger.info("Doing something...")
   ```

2. **新增 API 端點**
   ```python
   # app/api/main.py
   @app.post("/api/my-endpoint")
   async def my_endpoint(request: MyRequest):
       # 實作
       pass
   ```

3. **新增配置**
   ```python
   # app/config/settings.py
   class Settings(BaseSettings):
       my_setting: str = "default"
   ```

### 執行測試

```bash
# 健康檢查
curl http://localhost:8000/api/health

# 測試問答
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "測試問題"}'
```

## 🐛 疑難排解

### 問題：import 錯誤

**解決方法**：
```bash
# 確認在專案根目錄
pwd

# 重新安裝依賴
pip install -r requirements-new.txt
```

### 問題：API Key 未設定

**解決方法**：
1. 檢查 `.env` 檔案是否存在
2. 確認 `OPENAI_API_KEY` 已設定
3. 或通過 API 設定：
   ```bash
   curl -X POST http://localhost:8000/api/admin/api-key \
     -H "Content-Type: application/json" \
     -d '{"api_key": "sk-..."}'
   ```

### 問題：ChromaDB 無資料

**解決方法**：
```bash
# 重新初始化資料
python init_data.py
```

### 問題：端口被佔用

**解決方法**：
```bash
# 修改 .env 中的 API_PORT
# 或直接指定端口
API_PORT=8001 python run_api.py
```

## 🧪 測試

專案包含完整的單元測試套件，覆蓋率達 **74%**。

### 執行測試

```bash
# 執行所有測試（推薦）
python run_tests.py

# 或使用快速測試腳本
./quick_test.sh all           # 所有測試
./quick_test.sh api           # 只測試 API
./quick_test.sh services      # 只測試服務層
./quick_test.sh coverage      # 生成覆蓋率報告

# 使用 pytest 直接執行
pytest -v                     # 詳細輸出
pytest tests/test_api.py      # 測試特定檔案
pytest --cov=app              # 生成覆蓋率
```

### 測試覆蓋範圍

- ✅ **配置層**: 100% 覆蓋（6 個測試）
- ✅ **核心功能**: 96% 覆蓋（10 個測試）
- ✅ **資料模型**: 100% 覆蓋（15 個測試）
- ✅ **工具函數**: 100% 覆蓋（13 個測試）
- ✅ **服務層**: 75% 覆蓋（10 個測試）
- ✅ **API 層**: 51% 覆蓋（10 個測試）

**測試總計**: 64 個測試，全部通過 ✅

更多詳情請參閱：
- [測試文檔](../TESTING.md) - 測試指南和最佳實踐
- [測試報告](../TEST_REPORT.md) - 完整的測試結果和覆蓋率分析

## 📝 待辦事項

- [x] ✅ 新增單元測試（64 個測試，74% 覆蓋率）
- [ ] 新增整合測試
- [ ] Docker 支援
- [ ] CI/CD 配置
- [ ] 效能監控
- [ ] 多語言支援
- [ ] 前端重構（Vue.js / React）

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

## 📄 授權

MIT License

## 📮 聯絡資訊

如有問題，請透過 GitHub Issues 聯絡。

---

**享受使用重構後的 AirPods Q&A 系統！** 🎉
