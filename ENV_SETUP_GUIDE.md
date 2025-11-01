# 🔐 API Key 安全設定指南

## 為什麼使用 .env 文件？

- ✅ **安全**：不會提交到 Git
- ✅ **方便**：只需設定一次
- ✅ **標準**：業界標準做法
- ✅ **保護**：檔案權限設為 600（僅擁有者可讀寫）

---

## 📝 快速設定（推薦）

### 方式 1：使用自動設置腳本

```bash
./setup_env.sh
```

按照提示輸入您的 OpenAI API Key，腳本會自動：
1. 創建 `.env` 檔案
2. 設定正確的檔案權限
3. 驗證 API Key 格式

---

### 方式 2：手動創建

#### 步驟 1：創建 .env 檔案

在專案根目錄執行：

```bash
cat > .env << 'EOF'
# OpenAI API Key
OPENAI_API_KEY=sk-your-actual-api-key-here

# Google Gemini API Key (選用)
# GEMINI_API_KEY=your-gemini-key-here

# 服務器設定
HOST=0.0.0.0
PORT=8000
FRONTEND_PORT=8080

# 資料庫設定
DB_PATH=./web/backend/chroma_db
COLLECTION_NAME=text_embedding_openai
EOF
```

#### 步驟 2：編輯 API Key

```bash
nano .env  # 或使用您喜歡的編輯器
```

將 `sk-your-actual-api-key-here` 替換為您的真實 API Key。

#### 步驟 3：設定檔案權限

```bash
chmod 600 .env
```

這確保只有您可以讀寫此檔案。

---

## 🔍 驗證設定

### 檢查 .env 文件是否存在

```bash
ls -la .env
```

應該顯示：`-rw-------`（僅擁有者可讀寫）

### 測試 API Key 載入

```bash
python3 << 'EOF'
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    print(f"✅ API Key 已正確載入: {api_key[:10]}...")
else:
    print("❌ API Key 未設定")
EOF
```

---

## 🚀 啟動系統

設定完成後，直接啟動：

```bash
./start_all.sh
```

後端啟動時會顯示：
```
✅ API Key 已載入 (開頭: sk-proj-QM...)
```

---

## 🔒 安全最佳實踐

### 1. 檔案權限

確保 `.env` 權限正確：

```bash
# 檢查權限
ls -l .env

# 應該顯示：-rw-------
# 如果不是，執行：
chmod 600 .env
```

### 2. Git 忽略

確認 `.gitignore` 包含 `.env`：

```bash
grep -q "^\.env$" .gitignore && echo "✅ .env 已被忽略" || echo "❌ 請加入 .gitignore"
```

### 3. 定期檢查

```bash
# 確保 .env 未被追蹤
git status | grep .env

# 如果出現，立即執行：
git rm --cached .env
git commit -m "Remove .env from tracking"
```

### 4. API Key 輪換

建議定期更換 API Key：

1. 前往 [OpenAI API Keys](https://platform.openai.com/api-keys)
2. 撤銷舊的 Key
3. 創建新的 Key
4. 更新 `.env` 文件

---

## 🛠️ 疑難排解

### 問題：找不到 .env 檔案

```bash
# 檢查當前目錄
pwd

# 應該在專案根目錄，如果不是：
cd /path/to/grad-project
```

### 問題：權限錯誤

```bash
chmod 600 .env
```

### 問題：API Key 未載入

```bash
# 檢查 .env 格式
cat .env

# 確保格式正確：
# OPENAI_API_KEY=sk-proj-xxx（不要有空格）
```

### 問題：後端顯示警告

如果看到：
```
⚠️  警告：OPENAI_API_KEY 未設定！
```

檢查：
1. `.env` 檔案是否在專案根目錄
2. 格式是否正確（`OPENAI_API_KEY=sk-...`，沒有多餘空格）
3. 是否有執行 `source` 或重啟終端

---

## 📋 .env 檔案範本

```env
# OpenAI API Key（必需）
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx

# Google Gemini API Key（選用）
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxx

# 服務器設定
HOST=0.0.0.0
PORT=8000
FRONTEND_PORT=8080

# 資料庫設定
DB_PATH=./web/backend/chroma_db
COLLECTION_NAME=text_embedding_openai

# 除錯模式（選用）
DEBUG=false
LOG_LEVEL=INFO
```

---

## 🎯 檢查清單

設定完成後，確認以下項目：

- [ ] `.env` 檔案已創建在專案根目錄
- [ ] 檔案權限為 `-rw-------` (600)
- [ ] API Key 格式正確（以 `sk-` 開頭）
- [ ] `.env` 在 `.gitignore` 中
- [ ] 執行 `git status` 不顯示 `.env`
- [ ] 後端啟動顯示 "✅ API Key 已載入"

---

## 💡 其他設定方式

### 使用 macOS Keychain（更安全）

如果需要更高安全性：

```bash
# 儲存到 Keychain
security add-generic-password -a "$USER" -s "openai_api_key" -w "sk-your-key"

# 創建載入腳本
cat > load_env.sh << 'EOF'
#!/bin/bash
export OPENAI_API_KEY=$(security find-generic-password -a "$USER" -s "openai_api_key" -w)
./start_all.sh
EOF

chmod +x load_env.sh
```

### 環境變數（傳統方式）

在 `~/.zshrc` 或 `~/.bash_profile` 加入：

```bash
export OPENAI_API_KEY='sk-your-key'
```

然後：

```bash
source ~/.zshrc  # 或 source ~/.bash_profile
```

---

## ❓ 常見問題

**Q: .env 和環境變數哪個優先？**

A: 如果兩者都設定，`.env` 檔案的值會覆蓋環境變數。

**Q: 可以在 .env 中使用註解嗎？**

A: 可以，使用 `#` 開頭。

**Q: 需要重啟系統嗎？**

A: 不需要，只需重啟應用程式（`./stop_all.sh` 然後 `./start_all.sh`）。

**Q: 可以有多個 .env 文件嗎？**

A: 可以創建 `.env.local`、`.env.development` 等，但主要使用 `.env`。

---

**設定完成後，享受使用！** 🎉

