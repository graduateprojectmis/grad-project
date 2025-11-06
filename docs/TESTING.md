# 測試說明文檔

## 🧪 測試架構

本專案採用 **pytest** 作為測試框架，包含完整的單元測試和整合測試。

## 📁 測試結構

```
tests/
├── conftest.py              # 測試配置和共用 fixtures
├── test_config.py           # 配置管理測試
├── test_core.py             # 核心功能測試（日誌、例外）
├── test_models.py           # 資料模型測試
├── test_utils.py            # 工具函數測試
├── test_services.py         # 服務層測試（使用 Mock）
└── test_api.py              # API 端點測試
```

## 🚀 快速開始

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

#### 方式 2：直接使用 pytest

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

### 檢視覆蓋率報告

```bash
# 執行測試並生成覆蓋率報告
pytest --cov=app --cov-report=html

# 在瀏覽器中開啟報告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## 📊 測試覆蓋範圍

### 配置層測試 (`test_config.py`)
- ✅ Settings 單例模式
- ✅ 預設值驗證
- ✅ 環境變數載入
- ✅ 目錄自動建立
- ✅ CORS 設定
- ✅ 資料庫設定
- ✅ 日誌設定

### 核心層測試 (`test_core.py`)
- ✅ 日誌系統初始化
- ✅ 日誌檔案建立
- ✅ 日誌級別控制
- ✅ 自定義例外類別
- ✅ 例外繼承關係
- ✅ 例外訊息和詳情

### 資料模型測試 (`test_models.py`)
- ✅ 請求模型驗證
- ✅ 回應模型結構
- ✅ 欄位驗證規則
- ✅ 預設值設定
- ✅ 文件模型結構
- ✅ API Key 模型

### 工具函數測試 (`test_utils.py`)
- ✅ 文字清理
- ✅ 文字分割
- ✅ JSON 讀取
- ✅ JSON 儲存
- ✅ 中文處理
- ✅ 錯誤處理

### 服務層測試 (`test_services.py`)
- ✅ 資料庫服務初始化
- ✅ 資料插入
- ✅ 資料查詢
- ✅ 資料計數
- ✅ 資料清空
- ✅ 嵌入服務（使用 Mock）
- ✅ LLM 服務（使用 Mock）
- ✅ 錯誤處理

### API 測試 (`test_api.py`)
- ✅ 健康檢查端點
- ✅ 根路徑端點
- ✅ 問答端點（成功/失敗）
- ✅ 搜尋端點
- ✅ Admin API
- ✅ 請求驗證
- ✅ 錯誤回應

## 🎯 測試標記

使用標記來分類和選擇性執行測試：

```bash
# 只執行單元測試
pytest -m unit

# 只執行整合測試
pytest -m integration

# 執行 API 測試
pytest -m api

# 執行服務層測試
pytest -m service

# 跳過慢速測試
pytest -m "not slow"
```

## 🔧 測試配置

測試配置在 `pytest.ini` 中定義：

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

## 📝 編寫測試

### 測試檔案命名

- 檔案名稱以 `test_` 開頭
- 類別名稱以 `Test` 開頭
- 函數名稱以 `test_` 開頭

### 範例測試

```python
import pytest
from app.utils.text_processing import clean_text

class TestTextProcessing:
    """測試文字處理"""
    
    def test_clean_text_basic(self):
        """測試基本文字清理"""
        text = "  文字  "
        result = clean_text(text)
        assert result == "文字"
    
    def test_clean_text_empty(self):
        """測試空文字"""
        assert clean_text("") == ""
```

### 使用 Fixtures

```python
@pytest.fixture
def sample_data():
    """測試資料 fixture"""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """使用 fixture 的測試"""
    assert sample_data["key"] == "value"
```

### 使用 Mock

```python
from unittest.mock import patch, Mock

@patch('module.function')
def test_with_mock(mock_func):
    """使用 Mock 的測試"""
    mock_func.return_value = "mocked"
    result = function_to_test()
    assert result == "mocked"
```

## 🐛 除錯測試

### 使用 pdb

```python
def test_debug():
    import pdb; pdb.set_trace()
    # 在此設定斷點
    result = function_to_test()
    assert result == expected
```

### 顯示 print 輸出

```bash
pytest -s tests/test_file.py
```

### 詳細錯誤追蹤

```bash
pytest --tb=long
```

## 📈 持續整合

測試可以整合到 CI/CD 流程中：

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
          pip install -r requirements-new.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: pytest
```

## 🎓 最佳實踐

1. **每個測試應該獨立**
   - 不依賴其他測試的結果
   - 使用 fixtures 準備測試資料

2. **測試命名應該清楚**
   - 描述測試的目的
   - 例如：`test_clean_text_removes_whitespace`

3. **使用 AAA 模式**
   - Arrange（準備）
   - Act（執行）
   - Assert（驗證）

4. **測試邊界條件**
   - 空值
   - 極大/極小值
   - 錯誤輸入

5. **使用 Mock 隔離外部依賴**
   - API 呼叫
   - 資料庫操作
   - 檔案系統

## 📊 覆蓋率目標

- **整體覆蓋率**: ≥ 80%
- **核心模組**: ≥ 90%
- **工具函數**: ≥ 85%
- **API 端點**: ≥ 75%

## 🔗 相關資源

- [pytest 官方文檔](https://docs.pytest.org/)
- [pytest-cov 文檔](https://pytest-cov.readthedocs.io/)
- [unittest.mock 文檔](https://docs.python.org/3/library/unittest.mock.html)

---

**測試是確保程式碼品質的關鍵！** 🧪
