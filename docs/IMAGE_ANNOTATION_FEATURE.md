# 圖片標記功能使用指南

## 功能概述

這個功能允許使用者在上傳圖片時選擇是否啟用 AI 圖片標記功能。當啟用時，系統會使用 Google Gemini API 自動偵測圖片中的物件並標註邊界框。

## 前置需求

### 1. Google API Key

在 `.env` 檔案中設定 Google API Key：

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. 安裝相關套件

確保已安裝 Google Generative AI 套件：

```bash
pip install google-generativeai pillow
```

## 功能使用

### 1. 上傳圖片

1. 點擊輸入框左側的 📎 圖示選擇圖片
2. 支援的格式：JPG、PNG、GIF、WEBP
3. 檔案大小限制：5MB

### 2. 啟用圖片標記

當選擇圖片後，會顯示圖片預覽區域，包含：

- **圖片預覽**：顯示選擇的圖片縮圖
- **圖片資訊**：檔案名稱和大小
- **啟用圖片標記**：開關按鈕（點擊啟用/停用）
- **偵測目標**：當啟用標記時，可以指定要偵測的物件類型

### 3. 設定偵測目標

當啟用圖片標記後，可以輸入想要偵測的物件類型，例如：

- `person` - 偵測人物
- `car` - 偵測汽車
- `objects` - 偵測一般物件（預設值）
- `dog` - 偵測狗
- `chair` - 偵測椅子
- 其他任何你想偵測的物件

### 4. 發送與處理

點擊「發送」按鈕後：

#### 如果**未啟用**圖片標記：
- 圖片會被上傳到伺服器
- 系統回應上傳成功訊息，包含檔案資訊

#### 如果**啟用**圖片標記：
- 圖片會被上傳並進行 AI 分析
- 系統會偵測圖片中的目標物件
- 回應包含：
  - 偵測到的物件數量
  - 每個物件的標籤和座標
  - 標註圖片的儲存路徑

## API 端點

### 1. 上傳圖片（不標註）

```
POST /api/upload
Content-Type: multipart/form-data

Body:
- file: 圖片檔案
```

### 2. 上傳並標註圖片

```
POST /api/annotate-image
Content-Type: multipart/form-data

Body:
- file: 圖片檔案
- target_item: 偵測目標（字串，預設為 "objects"）
```

**回應範例：**

```json
{
  "status": "success",
  "message": "成功偵測並標註 3 個物件",
  "total_detected": 3,
  "objects": [
    {
      "box_2d": [100, 200, 300, 400],
      "label": "person"
    },
    {
      "box_2d": [150, 250, 350, 450],
      "label": "chair"
    },
    {
      "box_2d": [200, 300, 400, 500],
      "label": "table"
    }
  ],
  "annotated_images": [
    "/path/to/person_0_detected.png",
    "/path/to/chair_1_detected.png",
    "/path/to/table_2_detected.png"
  ]
}
```

## 技術架構

### 後端

1. **Schema 定義** (`app/models/schemas.py`)
   - `ImageAnnotationRequest`: 圖片標註請求
   - `ImageAnnotationResponse`: 圖片標註回應
   - `DetectedObjectResponse`: 偵測物件資訊

2. **API 端點** (`app/api/main.py`)
   - `/api/upload`: 單純上傳圖片
   - `/api/annotate-image`: 上傳並標註圖片

3. **標註服務** (`app/services/annotating_service.py`)
   - `ImageAnnotationService`: 使用 Google Gemini 進行物件偵測
   - `detect_objects()`: 偵測物件
   - `annotate_image()`: 標註圖片並儲存

### 前端

1. **API 服務** (`frontend-react/src/services/api.js`)
   - `uploadImage()`: 上傳圖片
   - `annotateImage()`: 標註圖片

2. **UI 組件** (`frontend-react/src/components/InputArea.jsx`)
   - 圖片預覽
   - 標記開關
   - 偵測目標輸入框
   - 上傳邏輯處理

## 輸出檔案

標註後的圖片會儲存在：

```
data/output/Annotated_Image/
```

每個偵測到的物件會生成一個標註圖片，檔案命名格式：

```
{物件標籤}_{索引}_detected.png
```

例如：
- `person_0_detected.png`
- `chair_1_detected.png`

## 注意事項

1. **API Key 需求**：必須設定 Google API Key 才能使用標註功能
2. **圖片格式**：僅支援常見圖片格式
3. **檔案大小**：限制 5MB 以內
4. **偵測準確度**：依賴於 Google Gemini 模型的能力
5. **處理時間**：標註需要一定時間，請耐心等待

## 故障排除

### 問題：無法標註圖片

**可能原因：**
1. 未設定 Google API Key
2. API Key 無效或已過期
3. 圖片格式不支援
4. 網路連線問題

**解決方法：**
1. 檢查 `.env` 檔案中的 `GOOGLE_API_KEY`
2. 確認 API Key 有效
3. 使用支援的圖片格式
4. 檢查網路連線

### 問題：標註結果不準確

**可能原因：**
1. 偵測目標描述不明確
2. 圖片品質不佳
3. 物件在圖片中不明顯

**解決方法：**
1. 使用更具體的偵測目標描述
2. 使用高解析度圖片
3. 確保物件在圖片中清晰可見

## 更新日誌

### v1.0.0 (2025-11-14)
- ✅ 新增圖片標記開關
- ✅ 支援自訂偵測目標
- ✅ 整合 Google Gemini API
- ✅ 自動儲存標註圖片
- ✅ 完整的錯誤處理
