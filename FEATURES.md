# ✨ 功能特色

## 🎯 核心功能

### 1. 智慧問答系統

- 🤖 **AI 驅動**：使用 OpenAI GPT-4o-mini
- 🔍 **語義搜尋**：基於 ChromaDB 向量資料庫
- 📚 **知識來源**：AirPods 完整說明書
- 💬 **自然對話**：友善的對話式界面

### 2. 網頁 API Key 管理 🆕

- 🌐 **網頁輸入**：直接在前端設定 API Key
- 💾 **自動儲存**：存入伺服器 `.env` 文件
- 🔐 **安全保護**：檔案權限 600
- ⚡ **即時生效**：無需重啟服務
- 📊 **狀態顯示**：實時查看 API Key 狀態

### 3. 現代化界面

- 🎨 **美觀設計**：現代化 UI/UX
- 📱 **響應式**：支援各種螢幕尺寸
- 🌙 **深色模式**：（計劃中）
- ⚡ **流暢動畫**：載入和狀態提示

### 4. 文件搜尋

- 🔍 **向量搜尋**：快速找到相關內容
- 📝 **準確匹配**：語義理解，非關鍵字匹配
- 🎯 **相關性排序**：最相關的結果優先

### 5. 圖片上傳

- 📎 **支援格式**：JPG, PNG, GIF, WEBP
- 📏 **大小限制**：最大 5MB
- 🖼️ **即時預覽**：上傳前預覽
- ✅ **格式驗證**：自動檢查檔案類型

---

## 🔒 安全功能

### 1. API Key 保護

- 🔐 **伺服器端儲存**：不在前端暴露
- 🔑 **權限控制**：檔案權限 600
- 🚫 **Git 忽略**：自動排除版本控制
- 🔄 **隨時更新**：可隨時更換

### 2. CORS 設定

- 🌐 **跨域控制**：僅允許指定來源
- ✅ **本地開發**：localhost 預設允許
- 🔒 **生產環境**：可設定允許的域名

### 3. 輸入驗證

- ✅ **前端驗證**：即時回饋
- ✅ **後端驗證**：雙重保護
- 🛡️ **防注入**：安全的資料處理

---

## ⚡ 效能優化

### 1. 非同步處理

- 🚀 **FastAPI**：高效能後端框架
- ⚡ **uvloop**：加速 asyncio（macOS/Linux）
- 🔄 **並行處理**：支援多個請求

### 2. 快取機制

- 💾 **向量快取**：ChromaDB 持久化
- 🗃️ **資料預載**：啟動時載入資料
- ⚡ **快速回應**：優化查詢速度

### 3. 輕量前端

- 📦 **無框架**：純 JavaScript
- 🎯 **按需載入**：減少初始載入
- 🌐 **靜態服務**：Python HTTP server

---

## 🛠️ 開發者功能

### 1. RESTful API

```
GET  /api/health              # 健康檢查
POST /api/ask                 # 智慧問答
POST /api/search              # 文件搜尋
POST /api/upload              # 圖片上傳
POST /api/config/apikey       # 儲存 API Key
GET  /api/config/apikey/status # 檢查狀態
DELETE /api/config/apikey     # 刪除 API Key
```

### 2. API 文檔

- 📚 **Swagger UI**：http://localhost:8000/api/docs
- 📖 **ReDoc**：http://localhost:8000/api/redoc
- 🔧 **互動式測試**：直接在文檔中測試

### 3. 日誌系統

- 📝 **後端日誌**：`logs/backend.log`
- 🌐 **前端日誌**：`logs/frontend.log`
- 🔍 **即時監控**：`tail -f logs/*.log`

### 4. 開發工具

```bash
# 資料處理工具
tools/airpods_manual_fetch.py      # 下載說明書
tools/clean_data.py                # 清理資料
tools/generate_embedding_openai.py # 生成嵌入向量
tools/ChromaDB.py                  # 管理資料庫

# 初始化腳本
init_chromadb.py                   # 初始化 ChromaDB
setup_env.sh                       # 設定環境變數
```

---

## 📊 資料處理

### 1. 文字處理

- ✂️ **智慧分割**：RecursiveCharacterTextSplitter
- 📏 **適當大小**：chunk_size=500
- 🔗 **保持上下文**：chunk_overlap=50

### 2. 嵌入生成

- 🤖 **OpenAI Embeddings**：text-embedding-ada-002
- 🔢 **向量維度**：1536
- 💾 **持久儲存**：JSON + ChromaDB

### 3. 向量搜尋

- 🎯 **餘弦相似度**：找出最相關內容
- 📊 **可調結果數**：top_k 參數
- ⚡ **快速查詢**：ChromaDB 優化

---

## 🎨 UI/UX 特色

### 1. 對話界面

- 💬 **對話氣泡**：清晰區分用戶和 AI
- 👤 **頭像顯示**：視覺化身份
- ⏰ **載入動畫**：思考中提示
- 📜 **自動滾動**：保持在最新訊息

### 2. 狀態指示

- 🟢 **連線狀態**：實時顯示
- 📊 **資料筆數**：顯示資料庫大小
- ⚙️ **API Key 狀態**：已設定/未設定

### 3. 範例問題

- 💡 **快速開始**：預設問題按鈕
- 🎯 **常見場景**：涵蓋主要功能
- 📝 **自動填入**：點擊即用

---

## 🚀 部署功能

### 1. 一鍵啟動

```bash
./start_all.sh  # 啟動前後端
./stop_all.sh   # 停止所有服務
```

### 2. 進程管理

- 🔄 **背景執行**：獨立進程
- 📝 **PID 記錄**：追蹤進程 ID
- 🛑 **優雅停止**：正確清理資源

### 3. 自動化

- ✅ **環境檢查**：Python、依賴
- 📦 **自動安裝**：requirements.txt
- 🗃️ **資料初始化**：ChromaDB 設定
- 🌐 **瀏覽器啟動**：自動開啟（macOS）

---

## 📈 即將推出

### 計劃功能

- [ ] 🌙 深色模式
- [ ] 📱 PWA 支援
- [ ] 🔊 語音輸入
- [ ] 📥 對話匯出
- [ ] 🌍 多語言支援
- [ ] 🎨 自訂主題
- [ ] 📊 使用統計
- [ ] 🔔 通知系統

### 改進計劃

- [ ] ⚡ 更快的回應速度
- [ ] 🎯 更精準的答案
- [ ] 📚 更多知識來源
- [ ] 🔐 進階安全功能

---

## 🎓 技術棧

### 後端

- **框架**：FastAPI
- **資料庫**：ChromaDB
- **AI 模型**：OpenAI GPT-4o-mini
- **嵌入**：text-embedding-ada-002
- **異步**：uvicorn + uvloop

### 前端

- **技術**：純 JavaScript (Vanilla JS)
- **HTTP**：Fetch API
- **UI**：自定義 CSS
- **服務器**：Python HTTP server

### 工具

- **環境管理**：python-dotenv
- **文字處理**：langchain
- **資料處理**：pandas, numpy
- **網頁抓取**：beautifulsoup4

---

**持續更新中...** 🚀

有新功能建議？歡迎提出！

