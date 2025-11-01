# 🎧 AirPods Q&A 智慧問答系統

基於 RAG（Retrieval-Augmented Generation）架構的 AirPods 說明書智慧問答系統。

> 📱 **系統需求**：本專案專為 **macOS** 設計

## ✨ 功能特色

- 🤖 智慧問答：使用 LLM 生成準確的答案
- 🔍 語義搜尋：基於 ChromaDB 的向量資料庫
- 🎨 現代化介面：美觀的網頁前端
- 📊 多種嵌入模型支援：OpenAI / Google Gemini
- 🌐 RESTful API：完整的後端 API 服務
- ⚡ 高性能：使用 uvloop 加速

## 🚀 快速開始

### 前置需求

1. **macOS** 作業系統
2. **Python 3.8+** 
3. **OpenAI API Key** ([獲取 API Key](https://platform.openai.com/api-keys))

### 安裝 Python（如果尚未安裝）

```bash
# 使用 Homebrew 安裝
brew install python3

# 驗證安裝
python3 --version
```

### 快速啟動

#### 方式 A：網頁設定 API Key（最簡單 ⭐）

```bash
# 1. 賦予執行權限（首次）
chmod +x start_all.sh stop_all.sh

# 2. 啟動系統
./start_all.sh

# 3. 在網頁上設定 API Key
# 開啟 http://localhost:8080
# 點擊右上角設定按鈕（⚙️）
# 輸入 API Key 並儲存
```

> 🌐 **新功能**：現在可以直接在網頁上輸入 API Key，自動儲存到 `.env` 文件！

#### 方式 B：命令列設定（傳統方式）

**選項 1：使用設置腳本**
```bash
./setup_env.sh
```

**選項 2：手動創建**
```bash
echo "OPENAI_API_KEY=sk-your-actual-api-key-here" > .env
chmod 600 .env
```

然後啟動：
```bash
./start_all.sh
```

> 💡 **安全提示**：兩種方式都會將 API Key 儲存在伺服器的 `.env` 文件中（已被 Git 忽略）

系統會自動：
- ✅ 檢查環境
- ✅ 安裝依賴套件
- ✅ 初始化 ChromaDB（如果需要）
- ✅ 啟動後端（port 8000）
- ✅ 啟動前端（port 8080）
- ✅ 自動在瀏覽器開啟

## 🌐 訪問介面

啟動成功後，系統會自動在瀏覽器開啟，或手動訪問：

- **前端介面**：http://localhost:8080
- **後端 API**：http://localhost:8000
- **API 文檔**：http://localhost:8000/api/docs

## 🛑 停止服務

```bash
./stop_all.sh
```

或在執行 `start_all.sh` 的終端按 `Ctrl+C`

## 📋 詳細安裝步驟

### 1. 克隆專案

```bash
git clone <repository-url>
cd grad-project
```

### 2. 安裝依賴套件

```bash
pip3 install -r requirements.txt
```

### 3. 設定 API Key（使用 .env 文件）

**方式 A：使用自動設置腳本（推薦）**

```bash
./setup_env.sh
```

按照提示輸入您的 OpenAI API Key。

**方式 B：手動創建 .env 文件**

```bash
# 創建 .env 文件
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-actual-api-key-here
EOF

# 設定檔案權限（僅擁有者可讀寫）
chmod 600 .env
```

**驗證設定**

```bash
# 檢查檔案是否存在
ls -la .env

# 應該顯示：-rw------- （僅擁有者可讀寫）
```

> 📘 **詳細設定指南**：查看 [ENV_SETUP_GUIDE.md](ENV_SETUP_GUIDE.md)

### 4. 初始化資料庫（首次使用）

```bash
python3 init_chromadb.py
```

### 5. 啟動系統

```bash
./start_all.sh
```

## 📁 專案結構

```
grad-project/
├── web/                      # Web 應用
│   ├── backend/             # 後端 API
│   │   ├── api.py          # FastAPI 服務
│   │   └── chroma_db/      # ChromaDB 資料庫
│   └── frontend/            # 前端介面
│       ├── index.html      # 主頁面
│       ├── css/            # 樣式表
│       └── js/             # JavaScript
├── tools/                   # 工具腳本
│   ├── ChromaDB.py         # 資料庫管理
│   ├── query_with_llm.py   # LLM 查詢
│   └── ...                 # 其他工具
├── output/                  # 輸出資料
│   └── json/               # JSON 資料
├── logs/                    # 日誌檔案
├── start_all.sh            # 啟動腳本
├── stop_all.sh             # 停止腳本
├── init_chromadb.py        # 資料庫初始化
└── requirements.txt        # Python 依賴
```

## 🔌 API 使用

### 智慧問答（使用 LLM）

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何配對 AirPods？",
    "top_k": 1
  }'
```

### 文件搜尋（僅搜尋）

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "配對",
    "n_results": 1
  }'
```

### 健康檢查

```bash
curl http://localhost:8000/api/health
```

### 完整 API 文檔

訪問：http://localhost:8000/api/docs

## 🛠️ 開發工具

### 查看即時日誌

```bash
# 後端日誌
tail -f logs/backend.log

# 前端日誌
tail -f logs/frontend.log
```

### 重新初始化資料庫

```bash
rm -rf web/backend/chroma_db
python3 init_chromadb.py
```

### 更新依賴套件

```bash
pip3 install -r requirements.txt --upgrade
```

## 🐛 疑難排解

### 問題：找不到 OPENAI_API_KEY

**解決方法**：
```bash
# 檢查是否設定
echo $OPENAI_API_KEY

# 如果沒有，設定它
export OPENAI_API_KEY='sk-your-api-key'
```

### 問題：Port 被占用

**解決方法**：
```bash
# 停止所有服務
./stop_all.sh

# 或手動檢查並停止
lsof -ti:8000 | xargs kill -9
lsof -ti:8080 | xargs kill -9
```

### 問題：ChromaDB 資料不存在

**解決方法**：
```bash
python3 init_chromadb.py
```

### 問題：依賴套件錯誤

**解決方法**：
```bash
pip3 install -r requirements.txt
```

### 問題：權限錯誤

**解決方法**：
```bash
chmod +x start_all.sh stop_all.sh
```

### 查看詳細錯誤

```bash
# 查看後端錯誤
cat logs/backend.log

# 即時監控
tail -f logs/backend.log
```

## 🔧 進階設定

### 自訂 Port

編輯 `web/backend/api.py`：
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,  # 修改此處
        reload=True
    )
```

### 使用 Gemini API（選用）

設定 Gemini API Key：
```bash
export GEMINI_API_KEY='your-gemini-api-key'
```

## 📊 效能優化

### 使用 uvloop（已啟用）

本專案已使用 `uvloop` 來提升 asyncio 效能（僅 macOS/Linux）。

### 調整 LLM 參數

編輯 `tools/query_with_llm.py`：
```python
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",  # 可改為 gpt-4、gpt-3.5-turbo 等
    temperature=0.3,       # 調整創造性（0-1）
    max_tokens=500,        # 限制回覆長度
)
```

## 🧪 測試

### 測試 API

```bash
# 測試後端健康狀態
curl http://localhost:8000/api/health

# 測試問答功能
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "AirPods 如何充電？", "top_k": 1}'
```

## 📝 開發

### 資料處理工具

```bash
# 下載說明書
python3 tools/airpods_manual_fetch.py

# 清理資料
python3 tools/clean_data.py

# 生成嵌入向量
python3 tools/generate_embedding_openai.py

# 初始化 ChromaDB
python3 tools/ChromaDB.py
```

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

本專案為學術用途。

## 📧 聯絡

如有問題，請聯絡專案維護者。

---

**享受使用 AirPods Q&A 系統！** 🎉
