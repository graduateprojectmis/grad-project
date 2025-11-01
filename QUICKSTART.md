# ⚡ 快速開始

## 🚀 2 分鐘啟動指南

### 方式 A：網頁設定（最簡單 ⭐）

#### 1️⃣ 賦予執行權限（首次）

```bash
chmod +x start_all.sh stop_all.sh
```

#### 2️⃣ 啟動系統

```bash
./start_all.sh
```

#### 3️⃣ 在網頁上設定 API Key

1. 系統會自動開啟瀏覽器（或手動訪問 http://localhost:8080）
2. 點擊右上角的 **⚙️ 設定按鈕**
3. 輸入您的 OpenAI API Key（[獲取 Key](https://platform.openai.com/api-keys)）
4. 點擊 **「儲存到伺服器」**

✅ 完成！開始使用吧！

---

### 方式 B：命令列設定（傳統）

#### 1️⃣ 設定 API Key

```bash
./setup_env.sh
```

#### 2️⃣ 啟動系統

```bash
./start_all.sh
```

---

> 🌐 **新功能**：現在支援在網頁上直接設定 API Key，更加方便！

---

## 📋 系統需求

- ✅ macOS 作業系統
- ✅ Python 3.8+
- ✅ OpenAI API Key

---

## 🔍 驗證安裝

### 檢查 Python

```bash
python3 --version
```

如果未安裝：
```bash
brew install python3
```

### 檢查 .env 文件

```bash
cat .env
```

應該顯示：
```
OPENAI_API_KEY=sk-proj-xxxx...
```

---

## 🛑 停止服務

```bash
./stop_all.sh
```

或按 `Ctrl+C`

---

## 🌐 訪問點

| 服務 | 網址 | 說明 |
|-----|------|------|
| 前端 | http://localhost:8080 | Web 界面 |
| API | http://localhost:8000 | 後端服務 |
| 文檔 | http://localhost:8000/api/docs | API 文檔 |

---

## 💡 常用命令

```bash
# 啟動
./start_all.sh

# 停止
./stop_all.sh

# 重啟
./stop_all.sh && ./start_all.sh

# 查看日誌
tail -f logs/backend.log   # 後端
tail -f logs/frontend.log  # 前端

# 重新設定 API Key
./setup_env.sh

# 重新初始化資料庫
rm -rf web/backend/chroma_db && python3 init_chromadb.py
```

---

## ❓ 遇到問題？

### Port 被占用

```bash
# 清理 port
lsof -ti:8000 | xargs kill -9
lsof -ti:8080 | xargs kill -9
```

### API Key 錯誤

```bash
# 重新設定
./setup_env.sh
```

### ChromaDB 錯誤

```bash
# 重新初始化
python3 init_chromadb.py
```

---

## 📚 更多資訊

- **完整文檔**：[README.md](README.md)
- **API Key 設定**：[ENV_SETUP_GUIDE.md](ENV_SETUP_GUIDE.md)
- **更新紀錄**：[CHANGELOG_MAC.md](CHANGELOG_MAC.md)

---

**享受使用！** 🎉

