# 快速參考指南

## 🚀 一分鐘快速開始

```bash
# 1. 安裝依賴
pip install -r requirements-new.txt

# 2. 設定 API Key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 3. 初始化資料
python init_data.py

# 4. 啟動服務
python run_api.py
```

訪問 http://localhost:8000/api/docs 查看 API 文檔

## 📁 重要檔案位置

| 檔案/目錄 | 說明 |
|----------|------|
| `app/` | 主應用程式目錄 |
| `app/api/main.py` | API 主程式 |
| `app/config/settings.py` | 配置管理 |
| `app/services/` | 業務邏輯服務 |
| `.env` | 環境變數（需自行建立） |
| `init_data.py` | 資料初始化腳本 |
| `run_api.py` | API 啟動腳本 |
| `test_api.py` | API 測試腳本 |
| `manage_db.py` | 資料庫管理工具 |

## 🔧 常用命令

### 啟動服務

```bash
# 方式 1：使用腳本
python run_api.py

# 方式 2：使用 Uvicorn
uvicorn app.api.main:app --reload

# 方式 3：使用自動化腳本
./start-new.sh
```

### 資料管理

```bash
# 初始化資料
python init_data.py

# 查看資料庫狀態
python manage_db.py status

# 查詢資料庫
python manage_db.py query "配對 AirPods"

# 清空資料庫
python manage_db.py clear
```

### 測試

```bash
# 執行 API 測試
python test_api.py

# 測試特定端點
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "如何配對 AirPods？"}'
```

## 🔑 環境變數

在 `.env` 檔案中設定：

```bash
# 必須
OPENAI_API_KEY=sk-your-api-key-here

# 選用
GOOGLE_API_KEY=your-google-api-key
ADMIN_TOKEN=your-admin-token

# API 設定
API_HOST=0.0.0.0
API_PORT=8000

# 日誌
LOG_LEVEL=INFO
```

## 📡 API 端點

### 基本端點

```bash
# 健康檢查
GET /api/health

# 根路徑
GET /
```

### 核心功能

```bash
# 智慧問答
POST /api/ask
Content-Type: application/json
{
  "question": "你的問題",
  "top_k": 1
}

# 語義搜尋
POST /api/search
Content-Type: application/json
{
  "query": "搜尋關鍵字",
  "n_results": 3
}
```

### 管理端點

```bash
# 查詢 API Key 狀態
GET /api/admin/api-key/status

# 設定 API Key
POST /api/admin/api-key
Content-Type: application/json
{
  "api_key": "sk-..."
}

# 清除 API Key
DELETE /api/admin/api-key
```

## 🐍 程式碼範例

### 使用服務層

```python
from app.services import EmbeddingService, DatabaseService, LLMService
from app.config import get_settings

# 獲取設定
settings = get_settings()

# 初始化服務
embedding_service = EmbeddingService(provider="openai")
db_service = DatabaseService()
llm_service = LLMService()

# 生成嵌入向量
embeddings = embedding_service.generate_embedding(["文字1", "文字2"])

# 查詢資料庫
documents, distances = db_service.query(embeddings[0], n_results=5)

# 生成答案
answer = llm_service.generate_answer("問題", "上下文")
```

### 新增 API 端點

```python
# 在 app/api/main.py 中

from app.models import YourRequest, YourResponse

@app.post("/api/your-endpoint", response_model=YourResponse)
async def your_endpoint(request: YourRequest):
    try:
        # 你的邏輯
        result = process_request(request)
        return YourResponse(data=result, status="success")
    except Exception as e:
        logger.error(f"錯誤：{e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 新增服務

```python
# 在 app/services/ 中新增檔案

from app.core.logger import get_logger
from app.config import get_settings

logger = get_logger(__name__)

class YourService:
    def __init__(self):
        self.settings = get_settings()
        logger.info("YourService initialized")
    
    def do_something(self, data):
        logger.debug(f"Processing: {data}")
        # 你的邏輯
        return result
```

## 🔍 除錯技巧

### 查看日誌

```bash
# 即時查看日誌
tail -f logs/app.log

# 搜尋錯誤
grep ERROR logs/app.log

# 查看最近 50 行
tail -n 50 logs/app.log
```

### 設定除錯模式

```bash
# 在 .env 中
LOG_LEVEL=DEBUG

# 或直接在命令列
LOG_LEVEL=DEBUG python run_api.py
```

### 使用 Python 除錯器

```python
# 在程式碼中插入斷點
import pdb; pdb.set_trace()

# 或使用 IPython
from IPython import embed; embed()
```

## 🐛 常見問題

### Q: Import 錯誤

```bash
# 確保在專案根目錄
cd /path/to/Grad-Project

# 確認 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 重新安裝依賴
pip install -r requirements-new.txt
```

### Q: API Key 錯誤

```bash
# 檢查 .env 檔案
cat .env

# 或通過 API 設定
curl -X POST http://localhost:8000/api/admin/api-key \
  -H "Content-Type: application/json" \
  -d '{"api_key": "sk-..."}'
```

### Q: ChromaDB 無資料

```bash
# 重新初始化
rm -rf data/chroma_db
python init_data.py
```

### Q: 端口被佔用

```bash
# 檢查佔用端口的程式
lsof -i :8000

# 終止程式
kill -9 <PID>

# 或使用其他端口
API_PORT=8001 python run_api.py
```

## 📊 效能優化

### 批次處理

```python
# ✅ 好：批次處理
texts = ["text1", "text2", "text3"]
embeddings = embedding_service.generate_embedding(texts)

# ❌ 差：逐一處理
for text in texts:
    embedding = embedding_service.generate_embedding([text])
```

### 快取結果

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(param):
    # 昂貴的操作
    return result
```

### 非同步處理

```python
# 使用 FastAPI 的非同步支援
@app.post("/api/async-endpoint")
async def async_endpoint(request: Request):
    result = await async_operation()
    return result
```

## 🧪 測試

### 單元測試（未來）

```python
# tests/test_services.py
import pytest
from app.services import EmbeddingService

def test_embedding_service():
    service = EmbeddingService(provider="openai", api_key="test-key")
    # 測試邏輯
```

### 整合測試（未來）

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
```

## 📦 部署

### Docker（未來）

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-new.txt .
RUN pip install -r requirements-new.txt
COPY . .
CMD ["python", "run_api.py"]
```

### Docker Compose（未來）

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
```

## 🔐 安全性

### API Key 管理

- ✅ 使用 .env 檔案（不要提交到 Git）
- ✅ 設定 ADMIN_TOKEN 保護管理端點
- ✅ 使用 HTTPS（生產環境）
- ✅ 實作請求限流

### CORS 設定

```python
# 在 app/config/settings.py 中
api_cors_origins: list[str] = [
    "https://your-domain.com",
    "https://www.your-domain.com"
]
```

## 📈 監控

### 健康檢查

```bash
# 定期檢查服務健康
*/5 * * * * curl http://localhost:8000/api/health
```

### 日誌監控

```bash
# 使用 tail 監控
tail -f logs/app.log | grep ERROR

# 或使用專業工具（未來）
# - ELK Stack
# - Grafana Loki
# - Datadog
```

## 🎯 最佳實踐

1. **始終使用型別提示**
   ```python
   def function(param: str) -> int:
       return len(param)
   ```

2. **使用日誌而非 print**
   ```python
   logger.info("訊息")  # ✅
   print("訊息")        # ❌
   ```

3. **處理所有例外**
   ```python
   try:
       operation()
   except SpecificError as e:
       logger.error(f"錯誤：{e}")
       raise
   ```

4. **編寫清晰的文檔字串**
   ```python
   def function(param: str) -> str:
       """
       簡短描述。
       
       Args:
           param: 參數說明
           
       Returns:
           返回值說明
       """
       return param
   ```

## 🔗 有用的連結

- FastAPI 文檔: https://fastapi.tiangolo.com/
- Pydantic 文檔: https://docs.pydantic.dev/
- ChromaDB 文檔: https://docs.trychroma.com/
- OpenAI API: https://platform.openai.com/docs/

---

**需要幫助？** 查看完整文檔：
- `README-NEW.md` - 完整說明
- `ARCHITECTURE.md` - 架構設計
- `REFACTORING_SUMMARY.md` - 重構摘要
