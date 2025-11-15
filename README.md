# Grad-Project — 可擴展的影像標註與知識服務框架

這個專案是一個模組化的框架，原始目標為影像標註與相關服務（包含向量索引、LLM 介接、標註流程），但設計上可容易擴展成多種不同性質的系統，例如：

- 客服系統（Customer Service）
- 企業知識庫系統（Enterprise Knowledge Base / KB）
- 學習輔助系統（Educational / Tutoring Assistant）

重點：本框架把一般 AI 應用常見的構件拆成清晰模組（資料匯入、前處理、向量化/embedding、檢索、LLM 呼叫、標註/回傳介面），方便替換或升級任何一個部件以符合不同產品需求。

## 快速導覽

- 代碼入口：`run_api.py`, `main.py`, `run_tests.py`
- 主要模組：`app/services/`（業務邏輯）、`app/models/`（資料 schema）、`app/api/`（HTTP 介面）、`app/utils/`（工具函式）
- 文件與圖表：`docs/`（包含架構說明與 draw.io 檔案）

## 架構概觀

下方為系統架構圖與資料流程圖（你已用 draw.io 繪製並匯出 PNG）。

### 系統架構圖
![System Architecture](docs/image/System%20Architecture.svg)

（若想看或編輯原始 draw.io 檔案：`docs/drawio/System Architecture.drawio`）

### 資料流程圖 / 使用流程
![Data Flow Chart](docs/image/Data%20Flow%20Chart.svg)

（原始 draw.io 檔案：`docs/drawio/Data Flow Chart.drawio`）

## 設計要點（契約）

輸入/輸出與錯誤模式的簡短契約：

- 輸入：上傳的文件或影像 (binary / file path)、結構化 metadata（JSON）
- 輸出：已標註的結果（JSON）、向量索引條目、LLM 回覆（text / structured）
- 錯誤模式：檔案格式錯誤、外部服務不可用（向量 DB / LLM）、模型回傳逾時

成功條件：新的輸入能被成功匯入、產生 embedding、被檢索並由 LLM 給出合理回覆或標註。

## 可擴展的使用情境（範例）

1. 客服系統：
	- 資料來源：客服歷史紀錄、FAQ、SOP 文件
	- 變動點：將影像處理模組替換為文本/對話匯入流程，保留 embedding 與檢索 + LLM 回覆模組。

2. 企業知識庫：
	- 資料來源：文件庫（PDF/Office）、內部 Wiki
	- 變動點：新增文件爬蟲 / 匯入器、設定權限層級、企業向量索引設計（多租戶或命名空間）。

3. 學習輔助系統：
	- 資料來源：教科書章節、練習題、學生歷史紀錄
	- 變動點：加入教學策略模組（逐步提示、難度分級）、學生模型（tracking）與評量回饋。

每一種應用主要差異在於「資料匯入管線」與「服務化策略（可解釋性、權限、回覆風格）」；其餘共用的核心模組（embedding、檢索、LLM 呼叫）可複用。

## 快速開始（開發環境）

1. 建議 Python 版本：3.10+。建立虛擬環境並安裝依賴：

```bash
# macOS / zsh 範例
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 啟動本地 API（範例）

```bash
python run_api.py
```

3. 執行測試套件

```bash
python run_tests.py
```

（專案中也包含 `start-new.sh`, `start-react.sh` 可協助啟動前端與整合測試）

## 專案結構（摘要）

- `app/api/` — FastAPI / HTTP 介面（或其他 web entrypoints）
- `app/services/` — 服務層（annotating_service、embedding_service、llm_service、database_service）
- `app/models/` — Pydantic schemas 與模型
- `app/utils/` — 檔案操作、文字處理等 helper
- `docs/` — 設計文件與圖表（draw.io 原始檔與匯出圖）

欲了解更詳細的系統說明，請參考 `docs/ARCHITECTURE.md`。

## 如何擴展（實作要點）

1. 定義新的匯入器（ingestor）：把外部資料（例如對話、PDF、影像）轉為框架可處理的中介格式。
2. 加入或替換 embedding 後端（例如使用 OpenAI, Cohere, 或自建模型），並更新 `app/services/embedding_service.py`。
3. 調整檢索層（向量 DB 設計）：若需要分層權限或命名空間，請在 `database_service` 中新增相應邏輯。
4. 自訂 LLM 回覆策略：透過 `llm_service` 攔截 prompt 與回覆格式。

## 測試與品質門檻

- 測試入口：`tests/`（包含單元測試與整合測試範例）。
- 建議：為每個新增的外部整合（新的 embedding provider、vector DB、LLM）新增一組測試用例。

