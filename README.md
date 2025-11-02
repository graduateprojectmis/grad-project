# AirPods Q&A 智慧問答系統

## 快速開始

### 前置需求

- **Python 3.8+**
- **OpenAI API Key**

### 安裝 Python（如果尚未安裝）

```bash
# macOS - 使用 Homebrew
brew install python3

# 驗證安裝
python3 --version
```

### 快速啟動

```bash
# 1. 賦予執行權限（首次執行時）
chmod +x start_all.sh stop_all.sh

# 2. 啟動系統
./start_all.sh
```

系統會自動完成以下步驟：

- 檢查 Python 環境
- 安裝依賴套件
- 初始化 ChromaDB（如果需要）
- 啟動後端服務（port 8000）
- 啟動前端服務（port 8080）
- 自動在瀏覽器開啟

### 設定 API Key

1. 開啟 http://localhost:8080
2. 點擊右上角設定按鈕
3. 輸入您的 OpenAI API Key
4. 點擊「儲存到伺服器」

> **安全性**：API Key 會加密儲存在後端伺服器的環境變數檔案（`.env`）中，僅限本機可管理，不會暴露在前端。

---

## 訪問介面

啟動成功後，系統會自動在瀏覽器開啟，或手動訪問：

| 服務 | 網址 | 說明 |
|------|------|------|
| 前端介面 | http://localhost:8080 | 使用者問答介面 |
| 後端 API | http://localhost:8000 | RESTful API 服務 |
| API 文檔 | http://localhost:8000/api/docs | Swagger API 文檔 |

---

## 停止服務

```bash
# 方法一：使用停止腳本
./stop_all.sh

# 方法二：在執行 start_all.sh 的終端按 Ctrl+C
```

---

## 詳細安裝步驟

### 1. 安裝依賴套件

```bash
pip3 install -r requirements.txt
```

### 2. 初始化資料庫（首次使用）

```bash
python3 init_chromadb.py
```

### 3. 啟動系統

```bash
./start_all.sh
```

然後在網頁上設定 API Key（點擊右上角設定按鈕，輸入後點擊「儲存到伺服器」）。

---

## 專案結構

```
grad-project-3/
├── web/                      # Web 應用
│   ├── backend/             # 後端 API
│   │   ├── api.py          # FastAPI 服務
│   │   └── chroma_db/      # ChromaDB 資料庫
│   └── frontend/            # 前端介面
│       ├── index.html      # 主頁面
│       ├── css/            # 樣式表
│       └── js/             # JavaScript 檔案
├── tools/                   # 工具腳本
│   ├── ChromaDB.py         # 資料庫管理
│   ├── query_with_llm.py   # LLM 查詢工具
│   └── ...                 # 其他工具
├── output/                  # 輸出資料
│   └── json/               # JSON 格式資料
├── logs/                    # 系統日誌檔案
├── start_all.sh            # 啟動腳本
├── stop_all.sh             # 停止腳本
├── init_chromadb.py        # 資料庫初始化
└── requirements.txt        # Python 依賴套件
```

---

## 維護操作

### 重新初始化資料庫

如果需要清空並重建資料庫：

```bash
# 刪除現有資料庫
rm -rf web/backend/chroma_db

# 重新初始化
python3 init_chromadb.py
```

### 更新依賴套件

```bash
pip3 install -r requirements.txt --upgrade
```

---

## 疑難排解

### 問題：未設定 API Key

**症狀**：查詢時出現錯誤提示

**解決方法**：
1. 開啟網頁 http://localhost:8080
2. 點擊右上角設定按鈕
3. 輸入您的 OpenAI API Key
4. 點擊「儲存到伺服器」

> **提示**：API Key 會安全儲存在後端的 `.env` 檔案中，僅限本機可管理。

### 問題：ChromaDB 資料不存在

**症狀**：查詢時提示資料庫未初始化

**解決方法**：
```bash
python3 init_chromadb.py
```

### 問題：依賴套件錯誤

**症狀**：啟動時出現 `ModuleNotFoundError`

**解決方法**：
```bash
pip3 install -r requirements.txt
```

### 問題：權限錯誤

**症狀**：無法執行 `.sh` 腳本

**解決方法**：
```bash
chmod +x start_all.sh stop_all.sh
```