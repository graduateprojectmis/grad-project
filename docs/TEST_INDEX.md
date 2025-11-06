# 📋 測試文件索引

## 📁 測試相關文件

本專案包含完整的測試套件和文檔，以下是所有測試相關文件的索引：

### 🧪 測試代碼

| 檔案 | 說明 | 測試數量 |
|------|------|----------|
| [tests/conftest.py](tests/conftest.py) | 測試配置和共用 fixtures | - |
| [tests/test_config.py](tests/test_config.py) | 配置管理測試 | 6 |
| [tests/test_core.py](tests/test_core.py) | 核心功能測試（日誌、例外） | 10 |
| [tests/test_models.py](tests/test_models.py) | Pydantic 資料模型測試 | 15 |
| [tests/test_utils.py](tests/test_utils.py) | 工具函數測試 | 13 |
| [tests/test_services.py](tests/test_services.py) | 服務層測試（使用 Mock） | 10 |
| [tests/test_api.py](tests/test_api.py) | FastAPI 端點測試 | 10 |

**總計**: 64 個測試 ✅

### 📖 測試文檔

| 檔案 | 說明 | 用途 |
|------|------|------|
| [TESTING.md](TESTING.md) | 測試完整指南 | 學習如何執行和編寫測試 |
| [TEST_REPORT.md](TEST_REPORT.md) | 詳細測試報告 | 查看測試結果和覆蓋率 |
| [TEST_SUMMARY.md](TEST_SUMMARY.md) | 測試總結 | 快速了解測試狀態 |
| [TEST_INDEX.md](TEST_INDEX.md) | 本文件 | 測試文件導航 |

### 🔧 測試工具

| 檔案 | 說明 | 用途 |
|------|------|------|
| [pytest.ini](pytest.ini) | pytest 配置 | 測試執行配置 |
| [requirements-test.txt](requirements-test.txt) | 測試依賴 | 安裝測試套件 |
| [run_tests.py](run_tests.py) | Python 測試腳本 | 執行完整測試套件 |
| [quick_test.sh](quick_test.sh) | Bash 快速腳本 | 快速執行特定測試 |

### 📊 測試報告

| 檔案/目錄 | 說明 | 如何生成 |
|-----------|------|----------|
| `htmlcov/` | HTML 覆蓋率報告 | `./quick_test.sh coverage` |
| `htmlcov/index.html` | 覆蓋率主頁 | 在瀏覽器開啟查看 |
| `.coverage` | 覆蓋率資料 | pytest 自動生成 |

---

## 🚀 快速開始

### 1️⃣ 安裝測試依賴

```bash
pip install -r requirements-test.txt
```

### 2️⃣ 執行測試

```bash
# 方式 1: 使用 Python 腳本（推薦）
python run_tests.py

# 方式 2: 使用 Bash 腳本
./quick_test.sh all

# 方式 3: 直接使用 pytest
pytest -v
```

### 3️⃣ 查看覆蓋率

```bash
./quick_test.sh coverage
# 會自動生成並開啟 htmlcov/index.html
```

---

## 📚 文檔導航

### 新手入門
1. 閱讀 [TESTING.md](TESTING.md) - 了解測試基礎
2. 執行 `python run_tests.py` - 運行測試
3. 查看 [TEST_REPORT.md](TEST_REPORT.md) - 理解測試結果

### 進階使用
1. 閱讀測試代碼 - 學習如何編寫測試
2. 使用 [quick_test.sh](quick_test.sh) - 高效測試工作流
3. 查看 HTML 覆蓋率報告 - 發現未測試的代碼

### 開發者
1. 參考現有測試 - 遵循測試模式
2. 使用 Mock 和 Fixture - 隔離依賴
3. 維護高覆蓋率 - 目標 80%+

---

## 🎯 測試覆蓋率概覽

```
整體覆蓋率: 74%

模組覆蓋率:
  ✅ 配置層:    100% (settings.py)
  ✅ 核心層:     96% (logger.py, exceptions.py)
  ✅ 資料模型:  100% (schemas.py)
  ✅ 工具函數:  100% (text_processing.py)
  ⚠️  服務層:     75% (database, embedding, llm)
  ⚠️  API 層:     51% (main.py)
```

---

## 📋 測試清單

### ✅ 已完成
- [x] 配置管理測試
- [x] 日誌系統測試
- [x] 自定義例外測試
- [x] Pydantic 模型測試
- [x] 文字處理測試
- [x] 檔案操作測試
- [x] 資料庫服務測試（Mock）
- [x] 嵌入服務測試（Mock）
- [x] LLM 服務測試（Mock）
- [x] API 端點測試

### 🔄 進行中
- [ ] 提升 API 層覆蓋率至 70%+
- [ ] 增加 Gemini 嵌入服務測試
- [ ] 新增錯誤處理測試

### 📅 計劃中
- [ ] 整合測試（實際 ChromaDB）
- [ ] 端到端測試
- [ ] 效能測試
- [ ] 壓力測試
- [ ] 安全測試
- [ ] CI/CD 整合

---

## 🛠️ 測試命令速查

```bash
# 基本測試
python run_tests.py              # 完整測試套件
pytest -v                        # 詳細輸出
pytest -s                        # 顯示 print

# 分類測試
./quick_test.sh api              # API 測試
./quick_test.sh services         # 服務層測試
./quick_test.sh utils            # 工具測試

# 特定測試
pytest tests/test_api.py         # 測試單個檔案
pytest tests/test_api.py::TestHealthEndpoint  # 測試單個類別
pytest -k "test_health"          # 測試名稱匹配

# 覆蓋率
pytest --cov=app                 # 生成覆蓋率
pytest --cov=app --cov-report=html  # HTML 報告
./quick_test.sh coverage         # 生成並開啟

# 除錯
pytest -x                        # 遇到失敗停止
pytest --lf                      # 只執行失敗的測試
pytest --pdb                     # 遇到失敗進入 debugger

# 效能
pytest --durations=10            # 顯示最慢的 10 個測試
```

---

## 📞 需要幫助？

- **測試指南**: [TESTING.md](TESTING.md)
- **測試報告**: [TEST_REPORT.md](TEST_REPORT.md)
- **快速參考**: `./quick_test.sh help`
- **pytest 文檔**: https://docs.pytest.org/

---

## 📈 統計數據

| 指標 | 數值 |
|------|------|
| 測試總數 | 64 |
| 通過率 | 100% ✅ |
| 程式碼覆蓋率 | 74% |
| 執行時間 | 1.37 秒 |
| 測試檔案 | 7 個 |
| 測試框架 | pytest 8.3.4 |
| Python 版本 | 3.13.9 |

---

**最後更新**: 2025-01-18  
**版本**: 1.0.0  
**狀態**: ✅ 穩定
