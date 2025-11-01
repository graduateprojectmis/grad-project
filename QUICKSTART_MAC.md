# 🚀 快速啟動指南 (macOS)

## ⚡ 3 分鐘快速開始

### 1️⃣ 設定 API Key

```bash
export OPENAI_API_KEY='sk-your-actual-api-key-here'
```

### 2️⃣ 賦予執行權限

```bash
chmod +x start_all.sh stop_all.sh
```

### 3️⃣ 啟動系統

```bash
./start_all.sh
```

就這麼簡單！🎉 系統會自動在瀏覽器開啟 http://localhost:8080

---

## 📝 詳細步驟

### 前置需求

- macOS 作業系統
- Python 3.8+ （執行 `python3 --version` 檢查）
- OpenAI API Key

### 如果沒有 Python

```bash
# 安裝 Homebrew（如果沒有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安裝 Python
brew install python3
```

### 永久設定 API Key（推薦）

#### 如果使用 zsh（macOS Catalina+ 預設）

```bash
echo "export OPENAI_API_KEY='sk-your-api-key'" >> ~/.zshrc
source ~/.zshrc
```

#### 如果使用 bash

```bash
echo "export OPENAI_API_KEY='sk-your-api-key'" >> ~/.bash_profile
source ~/.bash_profile
```

### 驗證設定

```bash
echo $OPENAI_API_KEY
```

應該顯示您的 API Key。

---

## 🎯 常用命令

### 啟動系統

```bash
./start_all.sh
```

### 停止系統

```bash
./stop_all.sh
```

或按 `Ctrl+C`

### 查看日誌

```bash
# 後端日誌（即時）
tail -f logs/backend.log

# 前端日誌（即時）
tail -f logs/frontend.log
```

### 重新初始化資料庫

```bash
rm -rf web/backend/chroma_db
python3 init_chromadb.py
```

---

## 🌐 訪問點

| 服務 | 網址 |
|-----|------|
| 前端介面 | http://localhost:8080 |
| 後端 API | http://localhost:8000 |
| API 文檔 | http://localhost:8000/api/docs |

---

## ❓ 常見問題

### Q: 出現 "Permission denied" 錯誤

```bash
chmod +x start_all.sh stop_all.sh
```

### Q: Port 8000 或 8080 被占用

```bash
./stop_all.sh

# 或強制清理
lsof -ti:8000 | xargs kill -9
lsof -ti:8080 | xargs kill -9
```

### Q: 出現 "OPENAI_API_KEY not found" 錯誤

```bash
export OPENAI_API_KEY='sk-your-api-key'
```

### Q: ChromaDB 資料不存在

```bash
python3 init_chromadb.py
```

### Q: 依賴套件錯誤

```bash
pip3 install -r requirements.txt
```

---

## 💡 一鍵命令（複製即用）

### 完整初始化並啟動

```bash
# 設定 API Key（記得替換）
export OPENAI_API_KEY='sk-your-api-key-here'

# 賦予權限
chmod +x *.sh

# 安裝依賴
pip3 install -r requirements.txt

# 初始化資料庫
python3 init_chromadb.py

# 啟動系統
./start_all.sh
```

### 快速重啟

```bash
./stop_all.sh && ./start_all.sh
```

---

## 🔍 測試 API

### 使用 curl

```bash
# 健康檢查
curl http://localhost:8000/api/health

# 問答測試
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "如何配對 AirPods？", "top_k": 1}'
```

### 使用瀏覽器

直接訪問 API 文檔：http://localhost:8000/api/docs

可以在網頁上互動式測試所有 API。

---

## 📚 更多資訊

完整文檔請參考：[README.md](README.md)

---

**祝使用愉快！** 🎉

