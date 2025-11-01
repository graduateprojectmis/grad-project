# 🌐 網頁 API Key 設定指南

## ✨ 功能介紹

現在您可以直接在網頁上輸入 OpenAI API Key，系統會自動將其安全地儲存到伺服器的 `.env` 文件中！

### 特色

- ✅ **網頁輸入**：直接在前端界面輸入
- ✅ **伺服器儲存**：自動存入 `.env` 文件
- ✅ **安全保護**：檔案權限 600
- ✅ **即時生效**：無需重啟服務
- ✅ **狀態顯示**：實時顯示 API Key 狀態

---

## 🚀 使用方式

### 方法 1：網頁設定（推薦）

#### 步驟 1：啟動系統

```bash
./start_all.sh
```

#### 步驟 2：開啟網頁

在瀏覽器訪問：http://localhost:8080

#### 步驟 3：設定 API Key

1. 點擊右上角的 **⚙️ 設定按鈕**
2. 在彈出的視窗中輸入您的 OpenAI API Key
3. 點擊 **「儲存到伺服器」** 按鈕

✅ 完成！API Key 已自動儲存到 `.env` 文件

#### 步驟 4：開始使用

直接在對話框輸入問題，系統會使用剛才設定的 API Key。

---

### 方法 2：命令列設定（傳統方式）

如果您更喜歡命令列：

```bash
# 使用設置腳本
./setup_env.sh

# 或手動創建
echo "OPENAI_API_KEY=sk-your-api-key" > .env
chmod 600 .env
```

---

## 🔐 安全機制

### 1. 伺服器端儲存

- API Key 儲存在伺服器的 `.env` 文件中
- **不會**儲存在前端（localStorage）
- **不會**在網頁上顯示完整的 Key

### 2. 檔案權限保護

```bash
# 自動設定為 600（僅擁有者可讀寫）
-rw-------  1 user  staff  .env
```

### 3. Git 保護

```gitignore
# .gitignore 已包含
.env
.env.local
```

確保 API Key 不會被提交到版本控制。

### 4. 傳輸安全

⚠️ **重要**：
- 在生產環境應使用 **HTTPS**
- 本地開發使用 HTTP 是安全的（localhost）

---

## 📋 API 端點

### 1. 儲存 API Key

```http
POST /api/config/apikey
Content-Type: application/json

{
  "api_key": "sk-your-openai-api-key"
}
```

**回應**：
```json
{
  "status": "success",
  "message": "API Key 已成功儲存到 .env 文件",
  "has_key": true
}
```

### 2. 檢查 API Key 狀態

```http
GET /api/config/apikey/status
```

**回應**：
```json
{
  "status": "success",
  "has_key": true,
  "key_preview": "sk-proj-QM..."
}
```

### 3. 刪除 API Key

```http
DELETE /api/config/apikey
```

**回應**：
```json
{
  "status": "success",
  "message": "API Key 已從 .env 文件中刪除",
  "has_key": false
}
```

---

## 🎯 工作流程

### 首次使用

```
1. 啟動系統 (./start_all.sh)
   ↓
2. 開啟網頁 (http://localhost:8080)
   ↓
3. 點擊設定按鈕 (⚙️)
   ↓
4. 輸入 API Key
   ↓
5. 點擊「儲存到伺服器」
   ↓
6. 開始使用！
```

### 日常使用

```
1. 啟動系統 (./start_all.sh)
   ↓
2. 開啟網頁 (http://localhost:8080)
   ↓
3. 直接開始對話
   （API Key 已自動載入）
```

---

## 🛠️ 技術細節

### 後端實現

**檔案**：`web/backend/api.py`

```python
@app.post("/api/config/apikey")
async def save_api_key(request: ApiKeyRequest):
    # 1. 驗證 API Key 格式
    # 2. 讀取或創建 .env 文件
    # 3. 更新 OPENAI_API_KEY
    # 4. 設定檔案權限 (600)
    # 5. 重新載入環境變數
    # 6. 更新 openai.api_key
```

### 前端實現

**檔案**：`web/frontend/js/app.js`

```javascript
async function saveApiKey() {
    // 1. 驗證輸入
    // 2. 發送 POST 請求到後端
    // 3. 處理回應
    // 4. 更新 UI 狀態
}
```

### 安全考量

1. **輸入驗證**
   - 前端：檢查格式（sk- 開頭）
   - 後端：再次驗證格式

2. **檔案權限**
   - 自動設定為 600
   - macOS/Linux 支援完整權限控制
   - Windows：盡力嘗試

3. **即時更新**
   - 儲存後立即重新載入環境變數
   - 動態更新 openai.api_key
   - 無需重啟服務

---

## 💡 最佳實踐

### 1. API Key 管理

```bash
# 定期檢查狀態
curl http://localhost:8000/api/config/apikey/status

# 查看 .env 內容（注意安全）
cat .env

# 檢查檔案權限
ls -la .env
```

### 2. 定期輪換

建議每 3-6 個月更換 API Key：

1. 前往 [OpenAI API Keys](https://platform.openai.com/api-keys)
2. 撤銷舊的 Key
3. 創建新的 Key
4. 在網頁上更新

### 3. 監控使用

```bash
# 查看後端日誌
tail -f logs/backend.log

# 檢查 API Key 載入狀態
# 應該看到：✅ API Key 已載入
```

---

## 🐛 疑難排解

### 問題：儲存失敗

**可能原因**：
- 檔案權限問題
- 路徑不正確
- 格式錯誤

**解決方法**：
```bash
# 檢查目錄權限
ls -ld .

# 手動創建 .env
touch .env
chmod 600 .env

# 再次嘗試在網頁上儲存
```

### 問題：API Key 不生效

**檢查步驟**：

1. 確認儲存成功
```bash
grep "OPENAI_API_KEY" .env
```

2. 重啟後端（如果需要）
```bash
./stop_all.sh
./start_all.sh
```

3. 檢查後端日誌
```bash
tail logs/backend.log
```

### 問題：無法連接到後端

**解決方法**：

```bash
# 檢查後端是否運行
lsof -ti:8000

# 重新啟動
./stop_all.sh && ./start_all.sh
```

---

## 🔄 比較：網頁設定 vs 命令列設定

| 特性 | 網頁設定 | 命令列設定 |
|-----|---------|-----------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 安全性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 即時生效 | ✅ 是 | ❌ 需重啟 |
| 狀態顯示 | ✅ 實時 | ❌ 手動查看 |
| 適合用戶 | 一般用戶 | 開發者 |

**推薦**：兩種方式都很安全，選擇您喜歡的即可！

---

## 📚 相關文檔

- **完整文檔**：[README.md](README.md)
- **快速開始**：[QUICKSTART.md](QUICKSTART.md)
- **命令列設定**：[ENV_SETUP_GUIDE.md](ENV_SETUP_GUIDE.md)

---

## ⚠️ 重要提醒

1. **只在本地使用**
   - 此功能設計用於本地開發
   - 生產環境應使用更嚴格的安全措施

2. **保護您的 API Key**
   - 不要分享給他人
   - 定期輪換
   - 監控使用量

3. **備份但加密**
   - 如需備份，請加密
   - 不要上傳到雲端

---

**現在您可以輕鬆地在網頁上管理 API Key 了！** 🎉

有任何問題，歡迎查看其他文檔或諮詢。

