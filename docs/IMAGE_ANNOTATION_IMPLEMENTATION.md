# 圖片標記功能實現總結

## 功能概述

已成功在前端新增圖片標記功能開關，並完整整合後端圖片標註服務。使用者可以選擇是否對上傳的圖片進行 AI 物件偵測和標註。

## 修改的檔案

### 後端修改

#### 1. `app/models/schemas.py`
**新增內容：**
- `ImageAnnotationRequest`: 圖片標註請求 Schema
- `ImageAnnotationResponse`: 圖片標註回應 Schema
- `DetectedObjectResponse`: 偵測物件資訊 Schema

#### 2. `app/api/main.py`
**新增內容：**
- 匯入 `ImageAnnotationService` 和相關 Schema
- 全域變數 `annotation_service`
- `/api/upload` 端點：單純上傳圖片
- `/api/annotate-image` 端點：上傳並標註圖片

**功能特點：**
- 支援多種圖片格式（JPG、PNG、GIF、WEBP）
- 完整的錯誤處理和驗證
- 自動初始化標註服務
- 回傳偵測結果和標註圖片路徑

### 前端修改

#### 3. `frontend-react/src/services/api.js`
**新增內容：**
- `annotateImage()` 函數：呼叫圖片標註 API

#### 4. `frontend-react/src/components/InputArea.jsx`
**新增內容：**
- 匯入 `Tag` 圖示和 `annotateImage` 函數
- 狀態管理：
  - `enableAnnotation`: 圖片標記開關狀態
  - `targetItem`: 偵測目標物件
- UI 組件：
  - 圖片標記開關（checkbox）
  - 偵測目標輸入框
- 邏輯處理：
  - 根據開關決定呼叫 `uploadImage` 或 `annotateImage`
  - 顯示標註結果訊息

#### 5. `frontend-react/src/components/InputArea.css`
**新增內容：**
- `.annotation-controls`: 標註控制區塊樣式
- `.annotation-toggle`: 開關按鈕樣式
- `.target-item-input`: 目標輸入框樣式
- 響應式設計和動畫效果

### 文件修改

#### 6. `docs/IMAGE_ANNOTATION_FEATURE.md` (新增)
**內容：**
- 功能使用指南
- API 端點說明
- 技術架構介紹
- 故障排除指南

#### 7. `.env.example` (新增)
**內容：**
- 環境變數範例
- 包含 GOOGLE_API_KEY 說明

#### 8. `test_image_annotation.py` (新增)
**內容：**
- 服務初始化測試
- 物件偵測測試
- 完整的測試流程

## 功能流程

### 使用者操作流程

```
1. 點擊 📎 按鈕選擇圖片
   ↓
2. 圖片預覽顯示
   ↓
3. (可選) 啟用「圖片標記」開關
   ↓
4. (如果啟用) 輸入偵測目標（預設：objects）
   ↓
5. 點擊「發送」按鈕
   ↓
6a. 如果未啟用標記：
    → 上傳圖片 → 顯示上傳成功訊息
    
6b. 如果啟用標記：
    → 上傳並標註 → 偵測物件 → 繪製邊界框
    → 儲存標註圖片 → 顯示偵測結果
```

### API 呼叫流程

```
前端
  ↓
InputArea.jsx (handleSubmit)
  ↓
enableAnnotation == true?
  ↓ Yes                    ↓ No
annotateImage()         uploadImage()
  ↓                        ↓
/api/annotate-image    /api/upload
  ↓
ImageAnnotationService
  ↓
Google Gemini API
  ↓
detect_objects() → annotate_image()
  ↓
儲存標註圖片
  ↓
回傳結果給前端
```

## 技術實現細節

### 後端技術

1. **Google Gemini API 整合**
   - 使用 `gemini-2.0-flash-exp` 模型
   - 支援物件偵測（Bounding Box）
   - 自動解析 JSON 回應

2. **圖片處理**
   - PIL (Pillow) 進行圖片載入和調整
   - 自動調整圖片大小（最大 1024x1024）
   - 繪製邊界框和標籤

3. **檔案管理**
   - 上傳圖片儲存至 `data/uploads/`
   - 標註圖片儲存至 `data/output/Annotated_Image/`
   - 檔案命名格式：`{label}_{index}_detected.png`

### 前端技術

1. **React Hooks**
   - `useState` 管理開關和輸入狀態
   - `useRef` 管理檔案輸入元素

2. **UI/UX 設計**
   - 開關按鈕使用原生 checkbox
   - 條件式顯示偵測目標輸入框
   - 平滑動畫和過渡效果

3. **錯誤處理**
   - 檔案類型驗證
   - 檔案大小限制（5MB）
   - API 錯誤回饋

## 設定說明

### 環境變數

在 `.env` 檔案中設定：

```bash
# Google API Key（必需用於圖片標註功能）
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI API Key（用於問答功能）
OPENAI_API_KEY=your_openai_api_key_here
```

### 目錄結構

```
data/
  ├── uploads/              # 上傳的原始圖片
  └── output/
      └── Annotated_Image/  # 標註後的圖片
```

## 測試方法

### 方法 1：使用測試腳本

```bash
# 測試服務初始化
python test_image_annotation.py

# 測試圖片偵測
python test_image_annotation.py ./test_image.jpg person
```

### 方法 2：使用前端介面

1. 啟動後端：`python run_api.py`
2. 啟動前端：`cd frontend-react && npm run dev`
3. 開啟瀏覽器訪問前端
4. 上傳圖片並測試標記功能

### 方法 3：使用 API 直接測試

```bash
# 測試上傳（不標註）
curl -X POST http://localhost:8000/api/upload \
  -F "file=@test_image.jpg"

# 測試標註
curl -X POST http://localhost:8000/api/annotate-image \
  -F "file=@test_image.jpg" \
  -F "target_item=person"
```

## API 文件

### POST /api/upload

上傳圖片（不進行標註）

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (圖片檔案)

**Response:**
```json
{
  "status": "success",
  "message": "圖片上傳成功",
  "filename": "example.jpg",
  "file_path": "/path/to/file",
  "file_size": 12345
}
```

### POST /api/annotate-image

上傳並標註圖片

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file` (圖片檔案)
  - `target_item` (偵測目標，預設：objects)

**Response:**
```json
{
  "status": "success",
  "message": "成功偵測並標註 3 個物件",
  "total_detected": 3,
  "objects": [
    {
      "box_2d": [100, 200, 300, 400],
      "label": "person"
    }
  ],
  "annotated_images": [
    "/path/to/person_0_detected.png"
  ]
}
```

## 已知限制

1. **API Key 需求**：必須設定 Google API Key
2. **圖片大小**：限制 5MB
3. **處理時間**：依網路速度和圖片複雜度而定
4. **偵測準確度**：依賴 Google Gemini 模型能力

## 未來改進建議

1. **批次處理**：支援多張圖片同時上傳和標註
2. **即時預覽**：在前端直接顯示標註結果圖片
3. **自訂邊界框樣式**：允許使用者選擇顏色和線條粗細
4. **匯出功能**：提供標註結果的 JSON 匯出
5. **歷史記錄**：保存標註歷史供查詢
6. **進度顯示**：顯示標註處理進度條

## 總結

✅ **完成項目：**
- 前端圖片標記開關 UI
- 後端圖片標註 API 端點
- Google Gemini 整合
- 完整的錯誤處理
- 文件和測試工具

✅ **功能特點：**
- 簡單易用的開關介面
- 靈活的偵測目標設定
- 自動儲存標註結果
- 完整的狀態回饋

✅ **代碼品質：**
- 清晰的代碼結構
- 完整的註解說明
- 適當的錯誤處理
- 響應式 UI 設計

這個功能已經完全整合到系統中，可以立即使用！
