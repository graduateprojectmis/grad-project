# 🧪 測試完整指南

> 完整的測試文檔、報告和使用說明

**測試框架**: pytest 8.3.4  
**測試覆蓋率**: 74%  
**測試總數**: 64 個  
**通過率**: 100% ✅

---

## 📋 目錄

1. [測試總覽](#測試總覽)
2. [快速開始](#快速開始)
3. [測試架構](#測試架構)
4. [測試結果](#測試結果)
5. [測試編寫](#測試編寫)
6. [覆蓋率報告](#覆蓋率報告)
7. [故障排除](#故障排除)

---

## 測試總覽

### 測試統計

```
┌─────────────────────────────────────────┐
│ 測試類型       │ 測試數量 │ 覆蓋率    │
├─────────────────────────────────────────┤
│ API 測試       │   10    │  51%     │
│ 配置測試       │    6    │ 100%     │
│ 核心功能測試    │   10    │  96%     │
│ 資料模型測試    │   15    │ 100%     │
│ 服務層測試     │   10    │  75%     │
│ 工具函數測試    │   13    │ 100%     │
├─────────────────────────────────────────┤
│ 總計          │   64    │  74%     │
└─────────────────────────────────────────┘
```

### 測試分布

- **單元測試**: 54 個 (84%)
- **整合測試**: 10 個 (16%)
- **執行時間**: 1.37 秒
- **平台**: macOS, Linux, Windows

---

## 快速開始

### 安裝測試依賴

```bash
pip install -r requirements-test.txt
```

或單獨安裝：

```bash
pip install pytest pytest-cov pytest-asyncio pytest-mock httpx faker
```

### 執行測試

#### 方式 1：使用測試腳本（推薦）

```bash
python run_tests.py
```

#### 方式 2：使用快速腳本

```bash
# 執行所有測試
./quick_test.sh all

# 分類測試
./quick_test.sh api          # API 層
./quick_test.sh services     # 服務層
./quick_test.sh utils        # 工具函數

# 生成覆蓋率報告
./quick_test.sh coverage
```

#### 方式 3：直接使用 pytest

```bash
# 執行所有測試
pytest

# 執行特定測試檔案
pytest tests/test_config.py

# 執行特定測試類別
pytest tests/test_config.py::TestSettings

# 執行特定測試函數
pytest tests/test_config.py::TestSettings::test_default_values

# 詳細輸出
pytest -v

# 顯示 print 輸出
pytest -s

# 停在第一個失敗
pytest -x

# 執行上次失敗的測試
pytest --lf
```

---

## 測試架構

### 📁 測試結構

```
tests/
├── conftest.py              # 測試配置和共用 fixtures
├── test_config.py           # 配置管理測試 (6 個)
├── test_core.py             # 核心功能測試 (10 個)
├── test_models.py           # 資料模型測試 (15 個)
├── test_utils.py            # 工具函數測試 (13 個)
├── test_services.py         # 服務層測試 (10 個)
└── test_api.py              # API 端點測試 (10 個)
```

### 測試配置

測試配置在 `pytest.ini` 中定義：

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

### Fixtures 說明

**conftest.py** 提供的共用 fixtures：

```python
@pytest.fixture
def settings():
    """測試用的設定"""
    return get_settings()

@pytest.fixture
def mock_db_path(tmp_path):
    """臨時資料庫路徑"""
    return str(tmp_path / "test_chroma_db")

@pytest.fixture
def client():
    """FastAPI 測試客戶端"""
    return TestClient(app)
```

---

## 測試結果

### 詳細測試報告

#### ✅ API 測試 (10 個測試)
- `test_health_check` - 健康檢查端點
- `test_root` - 根路徑端點
- `test_ask_question_success` - 問答成功場景
- `test_ask_empty_question` - 空問題驗證
- `test_ask_invalid_top_k` - 無效 top_k 參數驗證
- `test_search_success` - 搜尋成功場景
- `test_search_empty_query` - 空搜尋查詢驗證
- `test_get_api_key_status` - 獲取 API Key 狀態
- `test_set_api_key` - 設定 API Key
- `test_set_invalid_api_key` - 設定無效 API Key

#### ✅ 配置測試 (6 個測試)
- `test_settings_singleton` - Settings 單例模式
- `test_default_values` - 預設值驗證
- `test_data_directories_created` - 目錄自動建立
- `test_cors_origins` - CORS 設定
- `test_chroma_db_settings` - ChromaDB 設定
- `test_log_settings` - 日誌設定

#### ✅ 核心功能測試 (10 個測試)
- `test_setup_logger` - 日誌系統設定
- `test_get_logger` - 獲取 Logger
- `test_logger_file_creation` - 日誌檔案建立
- `test_logger_levels` - 日誌級別控制
- `test_app_exception` - 基礎例外類別
- `test_database_error` - 資料庫錯誤
- `test_embedding_error` - 嵌入錯誤
- `test_api_key_error` - API Key 錯誤
- `test_validation_error` - 驗證錯誤
- `test_exception_inheritance` - 例外繼承關係

#### ✅ 資料模型測試 (15 個測試)
- **QuestionRequest 模型** (4 個測試)
  - 有效請求、預設值、空問題、無效參數
- **QuestionResponse 模型** (2 個測試)
  - 有效回應、包含來源片段
- **SearchRequest 模型** (3 個測試)
  - 有效請求、預設值、無效參數
- **HealthResponse 模型** (1 個測試)
  - 有效回應結構
- **DocumentModels** (2 個測試)
  - DocumentChunk、DocumentWithEmbedding
- **APIKeyModels** (3 個測試)
  - 狀態回應、設定請求、空 Key 驗證

#### ✅ 服務層測試 (10 個測試)
- **DatabaseService** (6 個測試)
  - 初始化、插入、查詢、計數、清空、錯誤處理
- **EmbeddingService** (2 個測試)
  - OpenAI 嵌入、服務工廠
- **LLMService** (2 個測試)
  - 生成答案、生成摘要

#### ✅ 工具函數測試 (13 個測試)
- **TextProcessing** (8 個測試)
  - 基本清理、特殊字元、空文字、空白處理
  - 基本分割、長文字、短文字、空文字分割
- **FileOperations** (5 個測試)
  - 儲存與載入、不存在的檔案、巢狀路徑、中文處理、無效格式

### 程式碼覆蓋率

**整體覆蓋率: 74%**

| 模組 | 語句數 | 未覆蓋 | 覆蓋率 | 備註 |
|------|--------|--------|--------|------|
| `app/config/settings.py` | 36 | 0 | 100% | ✅ 完全覆蓋 |
| `app/core/exceptions.py` | 13 | 0 | 100% | ✅ 完全覆蓋 |
| `app/core/logger.py` | 24 | 1 | 96% | ⭐ 優秀 |
| `app/models/schemas.py` | 42 | 0 | 100% | ✅ 完全覆蓋 |
| `app/utils/text_processing.py` | 23 | 0 | 100% | ✅ 完全覆蓋 |
| `app/utils/file_operations.py` | 32 | 6 | 81% | ⚠️ 良好 |
| `app/services/llm_service.py` | 41 | 7 | 83% | ⚠️ 良好 |
| `app/services/database_service.py` | 64 | 16 | 75% | ⚠️ 良好 |
| `app/services/embedding_service.py` | 67 | 28 | 58% | ⚠️ 可改進 |
| `app/api/main.py` | 165 | 81 | 51% | ⚠️ 可改進 |
| **總計** | **525** | **139** | **74%** | |

### 執行結果

```
================================================ test session starts =================================================
platform darwin -- Python 3.13.9, pytest-8.3.4
collected 64 items

tests/test_api.py::TestHealthEndpoint::test_health_check PASSED                                    [  1%]
tests/test_api.py::TestRootEndpoint::test_root PASSED                                              [  3%]
tests/test_api.py::TestAskEndpoint::test_ask_question_success PASSED                               [  4%]
tests/test_api.py::TestAskEndpoint::test_ask_empty_question PASSED                                 [  6%]
tests/test_api.py::TestAskEndpoint::test_ask_invalid_top_k PASSED                                  [  7%]
tests/test_api.py::TestSearchEndpoint::test_search_success PASSED                                  [  9%]
tests/test_api.py::TestSearchEndpoint::test_search_empty_query PASSED                              [ 10%]
tests/test_api.py::TestAdminEndpoints::test_get_api_key_status PASSED                              [ 12%]
tests/test_api.py::TestAdminEndpoints::test_set_api_key PASSED                                     [ 14%]
tests/test_api.py::TestAdminEndpoints::test_set_invalid_api_key PASSED                             [ 15%]
tests/test_config.py::TestSettings::test_settings_singleton PASSED                                 [ 17%]
tests/test_config.py::TestSettings::test_default_values PASSED                                     [ 18%]
tests/test_config.py::TestSettings::test_data_directories_created PASSED                           [ 20%]
tests/test_config.py::TestSettings::test_cors_origins PASSED                                       [ 21%]
tests/test_config.py::TestSettings::test_chroma_db_settings PASSED                                 [ 23%]
tests/test_config.py::TestSettings::test_log_settings PASSED                                       [ 25%]
... (省略中間輸出) ...
tests/test_utils.py::TestFileOperations::test_load_json_invalid_format PASSED                      [100%]

=========================================== 64 passed, 1 warning in 1.37s ============================================

Coverage report:
Name                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------
app/__init__.py                         2      0   100%
app/api/__init__.py                     2      0   100%
app/api/main.py                       165     81    51%   
app/config/__init__.py                  2      0   100%
app/config/settings.py                 36      0   100%
app/core/__init__.py                    3      0   100%
app/core/exceptions.py                 13      0   100%
app/core/logger.py                     24      1    96%   
app/models/__init__.py                  2      0   100%
app/models/schemas.py                  42      0   100%
app/services/__init__.py                4      0   100%
app/services/database_service.py       64     16    75%   
app/services/embedding_service.py      67     28    58%   
app/services/llm_service.py            41      7    83%   
app/utils/__init__.py                   3      0   100%
app/utils/file_operations.py           32      6    81%   
app/utils/text_processing.py           23      0   100%
-----------------------------------------------------------------
TOTAL                                 525    139    74%
```

---

## 測試編寫

### 測試檔案命名

- 檔案名稱以 `test_` 開頭
- 類別名稱以 `Test` 開頭
- 函數名稱以 `test_` 開頭

### 範例測試

### 範例測試

#### 基本測試

```python
import pytest
from app.utils.text_processing import clean_text

class TestTextProcessing:
    """測試文字處理"""
    
    def test_clean_text_basic(self):
        """測試基本文字清理"""
        # Arrange
        text = "  文字  "
        
        # Act
        result = clean_text(text)
        
        # Assert
        assert result == "文字"
    
    def test_clean_text_empty(self):
        """測試空文字"""
        assert clean_text("") == ""
```

#### 使用 Fixtures

```python
@pytest.fixture
def sample_data():
    """測試資料 fixture"""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """使用 fixture 的測試"""
    assert sample_data["key"] == "value"
```

#### 使用 Mock

```python
from unittest.mock import patch, Mock

@patch('app.services.embedding_service.openai.Embedding.create')
def test_with_mock(mock_create):
    """使用 Mock 的測試"""
    # 設定 Mock 行為
    mock_create.return_value = {
        'data': [{'embedding': [0.1, 0.2, 0.3]}]
    }
    
    # 測試程式碼
    service = EmbeddingService()
    result = service.generate_embedding("test")
    
    # 驗證
    assert len(result[0]) == 3
    mock_create.assert_called_once()
```

#### 測試異步函數

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """測試異步函數"""
    result = await async_function()
    assert result == expected_value
```

---

## 覆蓋率報告

### 生成覆蓋率報告

```bash
# 生成 HTML 報告
pytest --cov=app --cov-report=html

# 生成終端報告
pytest --cov=app --cov-report=term

# 生成 XML 報告（用於 CI）
pytest --cov=app --cov-report=xml
```

### 查看報告

```bash
# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html

# Windows
start htmlcov/index.html
```

### 覆蓋率目標

- **整體覆蓋率**: ≥ 80%
- **核心模組**: ≥ 90%
- **工具函數**: ≥ 85%
- **API 端點**: ≥ 75%

### 提升覆蓋率建議

#### 1. API 層（51% → 70%+）
- 增加錯誤處理測試
- 測試更多端點場景
- 測試驗證邏輯

#### 2. 嵌入服務（58% → 75%+）
- 完整測試 Gemini 實作
- 增加錯誤場景測試
- 測試批次處理

#### 3. 未來測試類型
- 整合測試（實際 ChromaDB）
- 端到端測試
- 效能測試
- 壓力測試

---

## 故障排除

### 常見問題

#### 問題：測試失敗

**檢查清單**:
```bash
# 1. 確認依賴已安裝
pip install -r requirements-test.txt

# 2. 確認測試環境變數
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 3. 清除緩存
rm -rf .pytest_cache __pycache__ tests/__pycache__
```

#### 問題：Import 錯誤

```bash
# 確保在專案根目錄
cd /path/to/Grad-Project

# 設定 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 重新安裝
pip install -e .
```

#### 問題：Mock 不工作

```python
# 確認 Mock 路徑正確
# 錯誤：@patch('openai.Embedding.create')
# 正確：@patch('app.services.embedding_service.openai.Embedding.create')
```

#### 問題：覆蓋率不準確

```bash
# 清除舊的覆蓋率資料
rm -f .coverage
rm -rf htmlcov

# 重新生成
pytest --cov=app --cov-report=html
```

### 除錯測試

#### 使用 pdb

```python
def test_debug():
    import pdb; pdb.set_trace()
    # 在此設定斷點
    result = function_to_test()
    assert result == expected
```

#### 顯示 print 輸出

```bash
pytest -s tests/test_file.py
```

#### 詳細錯誤追蹤

```bash
pytest --tb=long
pytest --tb=short
pytest --tb=line
```

#### 只執行失敗的測試

```bash
pytest --lf  # last-failed
pytest --ff  # failed-first
```

---

## 最佳實踐

### 1. 測試應該獨立

```python
# ✅ 好：每個測試獨立
def test_function_1():
    data = setup_data()
    result = test_it(data)
    assert result

def test_function_2():
    data = setup_data()
    result = test_it(data)
    assert result

# ❌ 差：測試互相依賴
test_result = None

def test_function_1():
    global test_result
    test_result = test_it()
    
def test_function_2():
    assert test_result  # 依賴 test_function_1
```

### 2. 使用清晰的命名

```python
# ✅ 好：清楚描述測試內容
def test_clean_text_removes_leading_and_trailing_whitespace():
    pass

# ❌ 差：命名不清楚
def test_1():
    pass
```

### 3. AAA 模式

```python
def test_example():
    # Arrange（準備）
    data = setup_test_data()
    service = MyService()
    
    # Act（執行）
    result = service.process(data)
    
    # Assert（驗證）
    assert result == expected_value
```

### 4. 測試邊界條件

```python
def test_function_with_boundary_values():
    # 空值
    assert function("") == ""
    
    # 極大值
    assert function("x" * 10000)
    
    # 極小值
    assert function("x")
    
    # 錯誤輸入
    with pytest.raises(ValueError):
        function(None)
```

### 5. 使用 Mock 隔離外部依賴

```python
# ✅ 好：使用 Mock
@patch('requests.get')
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {'data': 'test'}
    result = fetch_data()
    assert result == {'data': 'test'}

# ❌ 差：實際呼叫外部 API
def test_api_call():
    result = requests.get('https://api.example.com')  # 不穩定
    assert result.status_code == 200
```

---

## 持續整合

### GitHub Actions 範例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run tests
        run: |
          pytest --cov=app --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
        with:
          file: ./coverage.xml
```

---

## 快速命令參考

```bash
# 基本測試
pytest                           # 執行所有測試
pytest -v                        # 詳細輸出
pytest -s                        # 顯示 print
pytest -x                        # 遇到失敗停止
pytest --lf                      # 只執行失敗的測試

# 特定測試
pytest tests/test_api.py         # 測試單個檔案
pytest tests/test_api.py::TestHealthEndpoint  # 測試單個類別
pytest -k "health"               # 測試名稱匹配

# 覆蓋率
pytest --cov=app                 # 生成覆蓋率
pytest --cov=app --cov-report=html  # HTML 報告
./quick_test.sh coverage         # 使用腳本

# 除錯
pytest --pdb                     # 遇到失敗進入 debugger
pytest --tb=short                # 簡短的錯誤追蹤
pytest --durations=10            # 顯示最慢的 10 個測試

# 快速腳本
python run_tests.py              # 完整測試套件
./quick_test.sh all              # 所有測試
./quick_test.sh api              # API 測試
./quick_test.sh services         # 服務測試
```

---

## 相關資源

### 內部文檔
- [PROJECT_GUIDE.md](./PROJECT_GUIDE.md) - 專案完整指南
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 架構設計

### 外部資源
- [pytest 官方文檔](https://docs.pytest.org/)
- [pytest-cov 文檔](https://pytest-cov.readthedocs.io/)
- [unittest.mock 文檔](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)

---
