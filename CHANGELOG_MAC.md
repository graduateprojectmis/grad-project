# 🔄 macOS 專用版本更新紀錄

## 📅 更新日期：2025-11-01

### ✨ 主要變更

#### 1. **移除 Windows 支援**
- ❌ 刪除 `start_all.bat`（Windows 批次檔）
- ❌ 刪除 `stop_all.bat`（Windows 停止腳本）
- ❌ 刪除舊的 `start_backend.sh` 和 `start_frontend.sh`

#### 2. **恢復 uvloop 支援**
- ✅ 在 `requirements.txt` 中恢復 `uvloop==0.21.0`
- ✅ macOS 支援 uvloop，可提升 asyncio 效能

#### 3. **優化啟動腳本**
- ✅ `start_all.sh` 改為 macOS 專用
- ✅ 新增 API Key 檢查和提示
- ✅ 自動在瀏覽器開啟前端（使用 `open` 命令）
- ✅ 改進錯誤訊息和日誌提示
- ✅ 更好的進程管理

#### 4. **優化停止腳本**
- ✅ `stop_all.sh` 改進為 macOS 專用
- ✅ 更可靠的進程清理
- ✅ 更好的狀態回饋

#### 5. **修復 OpenAI API 調用**
- ✅ 修正 `tools/query_with_llm.py` 的 API 調用方式
- ✅ 正確使用 `openai==0.28.0` 版本語法
- ✅ 新增 API Key 檢查

#### 6. **更新文檔**
- ✅ 重寫 `README.md` 為 macOS 專用
- ✅ 新增 `QUICKSTART_MAC.md` 快速啟動指南
- ✅ 新增 `.gitignore` 檔案
- ✅ 改進所有文檔的 macOS 相關說明

### 📝 使用方式變更

#### 之前（Windows 和 Mac 通用）

```bash
# Windows
start_all.bat

# Mac/Linux
./start_all.sh
```

#### 現在（僅 Mac）

```bash
# 設定 API Key
export OPENAI_API_KEY='sk-your-api-key'

# 啟動（macOS 專用）
./start_all.sh
```

### 🎯 優勢

1. **更簡潔**：移除不必要的 Windows 支援程式碼
2. **更快速**：使用 uvloop 提升效能
3. **更友善**：自動在瀏覽器開啟
4. **更可靠**：更好的錯誤處理和提示
5. **更專注**：專為 macOS 優化

### 📋 檔案清單

#### 新增檔案
- `QUICKSTART_MAC.md` - Mac 快速啟動指南
- `.gitignore` - Git 忽略檔案設定
- `CHANGELOG_MAC.md` - 本更新紀錄

#### 刪除檔案
- `start_all.bat` - Windows 啟動腳本
- `stop_all.bat` - Windows 停止腳本
- `start_backend.sh` - 舊後端啟動腳本
- `start_frontend.sh` - 舊前端啟動腳本
- `test_setup.py` - 測試腳本（用戶刪除）
- `setup_and_start.bat` - Windows 設定腳本（用戶刪除）
- `QUICKSTART.md` - 通用快速啟動指南（用戶刪除）

#### 修改檔案
- `start_all.sh` - 改為 macOS 專用，大幅優化
- `stop_all.sh` - 改為 macOS 專用，改進清理邏輯
- `requirements.txt` - 恢復 uvloop
- `tools/query_with_llm.py` - 修正 OpenAI API 調用
- `README.md` - 重寫為 macOS 專用文檔

### 🔧 技術細節

#### OpenAI API 調用修正

**之前（錯誤）：**
```python
client_llm = openai.ChatCompletion(api_key=os.getenv("OPENAI_API_KEY"))
response = client_llm.create(...)
```

**現在（正確）：**
```python
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.ChatCompletion.create(...)
```

#### 啟動腳本改進

- 新增 macOS 檢測
- 新增 API Key 檢測和提示
- 改進錯誤處理
- 自動在瀏覽器開啟（使用 `open` 命令）
- 更詳細的日誌和狀態回饋

### 🚀 快速開始

```bash
# 1. 設定 API Key
export OPENAI_API_KEY='sk-your-api-key'

# 2. 賦予執行權限
chmod +x start_all.sh stop_all.sh

# 3. 啟動系統
./start_all.sh
```

### 📚 文檔資源

- **快速開始**：`QUICKSTART_MAC.md`
- **完整文檔**：`README.md`
- **更新紀錄**：本檔案

### ⚠️ 重要提示

1. **本版本僅支援 macOS**
2. **需要設定 OPENAI_API_KEY 環境變數**
3. **首次執行會自動初始化 ChromaDB**
4. **建議將 API Key 永久加入 shell 配置檔**

### 🎉 享受使用！

所有功能已經過測試和優化，專為 macOS 環境設計。

