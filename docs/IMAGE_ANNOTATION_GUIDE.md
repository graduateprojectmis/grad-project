# 📸 圖片標註功能完整指南

> 使用 Google Gemini AI 進行智慧物件偵測和圖片標註

**版本**: 1.0  
**最後更新**: 2025-11-14  
**狀態**: ✅ 生產就緒

---

## 📚 目錄

1. [功能概述](#功能概述)
2. [快速開始](#快速開始)
3. [使用說明](#使用說明)
4. [API 文檔](#api-文檔)
5. [技術架構](#技術架構)
6. [開發指南](#開發指南)
7. [故障排除](#故障排除)

---

## 功能概述

圖片標註功能允許使用者上傳圖片並使用 AI 自動偵測物件，繪製邊界框並標註標籤。

### 核心能力

- ✅ **AI 物件偵測** - 使用 Google Gemini 2.0 Flash 模型
- ✅ **自動標註** - 繪製邊界框和標籤
- ✅ **多物件支援** - 可偵測多個物件
- ✅ **自訂目標** - 指定要偵測的物件類型
- ✅ **結果儲存** - 自動儲存標註圖片

### 支援格式

- **圖片格式**: JPG, PNG, GIF, WEBP
- **檔案大小**: 最大 5MB
- **圖片尺寸**: 自動調整至 1024x1024

---

## 快速開始

### 1. 設定 Google API Key

```bash
# 編輯 .env 檔案
nano .env

# 新增以下內容
GOOGLE_API_KEY=your_google_api_key_here
```

**如何取得 API Key？**
1. 訪問 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 登入 Google 帳號
3. 點擊「Create API Key」
4. 複製並貼到 `.env` 檔案

### 2. 安裝依賴

```bash
pip install google-generativeai pillow
```

### 3. 啟動服務

```bash
# 啟動後端
python run_api.py

# 啟動前端（另一個終端）
cd frontend-react
npm run dev
```

### 4. 測試功能

```bash
# 使用範例腳本測試
python example_annotation.py

# 或使用 API
curl -X POST http://localhost:8000/api/annotate-image \
  -F "file=@test_image.jpg" \
  -F "target_item=person"
```

---

## 使用說明

### 前端介面使用

#### 1. 上傳圖片

1. 點擊輸入框左側的 📎 圖示
2. 選擇圖片檔案（JPG, PNG, GIF, WEBP）
3. 圖片預覽會顯示在輸入框下方

#### 2. 啟用圖片標記

在圖片預覽區域：
- 勾選「✅ 啟用圖片標記」開關
- 輸入偵測目標（例如：`person`, `car`, `objects`）
- 預設目標為 `objects`

#### 3. 發送處理

點擊「發送」按鈕後：

**如果未啟用標記**：
- 圖片會被上傳到伺服器
- 顯示上傳成功訊息

**如果啟用標記**：
- 圖片會被上傳並分析
- AI 偵測圖片中的物件
- 顯示偵測結果：
  - 偵測到的物件數量
  - 每個物件的標籤和座標
  - 標註圖片的儲存路徑

### 命令列使用

#### 範例 1：基本標註

```python
from app.services import ImageAnnotationService

# 初始化服務
service = ImageAnnotationService()

# 標註圖片
saved_files = service.annotate_image(
    image_path="data/uploads/photo.jpg",
    target_item="person"
)

print(f"已儲存 {len(saved_files)} 個標註圖片")
```

#### 範例 2：只獲取偵測資訊

```python
# 獲取偵測摘要（不儲存圖片）
summary = service.get_detection_summary(
    image_path="data/uploads/photo.jpg",
    target_item="car"
)

print(f"偵測到 {summary['total_detected']} 個物件")
for obj in summary['objects']:
    print(f"- {obj['label']}: {obj['box_2d']}")
```

#### 範例 3：僅偵測物件

```python
# 只執行偵測，不標註和儲存
detected_objects = service.detect_objects(
    image_path="data/uploads/photo.jpg",
    target_item="objects"
)

for obj in detected_objects:
    print(f"{obj.label}: 座標 {obj.box_2d}")
```

### 偵測目標範例

| 目標 | 說明 | 使用場景 |
|------|------|----------|
| `person` | 人物 | 偵測照片中的人 |
| `car` | 汽車 | 街景中的車輛 |
| `dog` | 狗 | 寵物照片 |
| `cat` | 貓 | 寵物照片 |
| `chair` | 椅子 | 室內傢俱 |
| `table` | 桌子 | 室內傢俱 |
| `laptop` | 筆記型電腦 | 桌面物品 |
| `phone` | 手機 | 電子產品 |
| `tree` | 樹木 | 自然景物 |
| `objects` | 一般物件 | 偵測所有物件 |
| `button` | 按鈕 | UI 元素 |
| `icon` | 圖示 | UI 元素 |

---

## API 文檔

### 端點總覽

| 端點 | 方法 | 說明 |
|------|------|------|
| `/api/upload` | POST | 上傳圖片（不標註） |
| `/api/annotate-image` | POST | 上傳並標註圖片 |

### POST /api/upload

上傳圖片但不進行標註。

**請求**:
```http
POST /api/upload
Content-Type: multipart/form-data

Body:
  file: <image_file>
```

**回應**:
```json
{
  "status": "success",
  "message": "圖片上傳成功",
  "filename": "example.jpg",
  "file_path": "/data/uploads/example.jpg",
  "file_size": 123456
}
```

### POST /api/annotate-image

上傳圖片並進行 AI 標註。

**請求**:
```http
POST /api/annotate-image
Content-Type: multipart/form-data

Body:
  file: <image_file>
  target_item: "person"  (可選，預設 "objects")
```

**成功回應**:
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
    "/data/output/Annotated_Image/person_0_detected.png",
    "/data/output/Annotated_Image/chair_1_detected.png",
    "/data/output/Annotated_Image/table_2_detected.png"
  ]
}
```

**錯誤回應**:
```json
{
  "status": "error",
  "message": "錯誤訊息描述"
}
```

### 座標系統

邊界框座標格式：`[y_min, x_min, y_max, x_max]`

- 座標範圍：0-1000（正規化座標）
- 系統會自動轉換為實際像素座標
- 左上角為原點 (0, 0)

---

## 技術架構

### 後端架構

```
┌──────────────────────────────────────┐
│        FastAPI API Layer             │
│                                      │
│  /api/upload                         │
│  /api/annotate-image                 │
└──────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│    ImageAnnotationService            │
│                                      │
│  • detect_objects()                  │
│  • annotate_image()                  │
│  • get_detection_summary()           │
└──────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│      Google Gemini API               │
│                                      │
│  Model: gemini-2.0-flash-exp         │
│  Feature: Object Detection           │
└──────────────────────────────────────┘
```

### 前端架構

```
┌──────────────────────────────────────┐
│      React Components                │
│                                      │
│  • InputArea (上傳介面)              │
│  • ImagePreview (預覽)               │
│  • AnnotationToggle (開關)           │
└──────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│        API Service                   │
│                                      │
│  • uploadImage()                     │
│  • annotateImage()                   │
└──────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│     Backend API Endpoints            │
└──────────────────────────────────────┘
```

### 資料流程

```
使用者上傳圖片
    │
    ▼
前端：InputArea.jsx
    │
    ├─► 未啟用標記 ──► uploadImage() ──► /api/upload
    │                                        │
    │                                        ▼
    │                                    儲存檔案
    │                                        │
    │                                        ▼
    │                                    返回成功
    │
    └─► 啟用標記 ──► annotateImage() ──► /api/annotate-image
                                             │
                                             ▼
                                    ImageAnnotationService
                                             │
                                             ├─► 載入圖片
                                             ├─► 調整尺寸
                                             ├─► 呼叫 Gemini API
                                             ├─► 解析 JSON 回應
                                             ├─► 繪製邊界框
                                             └─► 儲存標註圖片
                                             │
                                             ▼
                                    返回偵測結果
```

### 檔案結構

```
app/
├── services/
│   └── annotating_service.py    # 圖片標註服務
│       ├── ImageAnnotationService
│       └── DetectedObject
│
├── models/
│   └── schemas.py               # API 資料模型
│       ├── ImageAnnotationRequest
│       ├── ImageAnnotationResponse
│       └── DetectedObjectResponse
│
└── api/
    └── main.py                  # API 端點
        ├── /api/upload
        └── /api/annotate-image

frontend-react/
├── src/
│   ├── components/
│   │   └── InputArea.jsx        # 上傳和標註 UI
│   └── services/
│       └── api.js               # API 呼叫封裝

data/
├── uploads/                     # 上傳的原始圖片
└── output/
    └── Annotated_Image/         # 標註後的圖片
```

---

## 開發指南

### 環境設定

```bash
# 1. 安裝 Python 依賴
pip install google-generativeai pillow

# 2. 設定 API Key
echo "GOOGLE_API_KEY=your-key-here" >> .env

# 3. 測試環境
python -c "import google.generativeai as genai; print('OK')"
```

### 服務初始化

```python
from app.services import ImageAnnotationService

# 方式 1：使用環境變數中的 API Key
service = ImageAnnotationService()

# 方式 2：手動指定 API Key
service = ImageAnnotationService(api_key="your-key")

# 方式 3：自訂參數
service = ImageAnnotationService(
    api_key="your-key",
    model="gemini-2.0-flash-exp",
    max_image_size=(2048, 2048)
)
```

### 整合到自己的應用

#### 1. 只使用偵測功能

```python
from app.services import ImageAnnotationService

service = ImageAnnotationService()

# 偵測物件
objects = service.detect_objects(
    image_path="your_image.jpg",
    target_item="person"
)

# 處理結果
for obj in objects:
    y_min, x_min, y_max, x_max = obj.box_2d
    print(f"偵測到 {obj.label} 在 ({x_min}, {y_min}) - ({x_max}, {y_max})")
```

#### 2. 標註並儲存

```python
# 標註圖片並儲存
saved_files = service.annotate_image(
    image_path="your_image.jpg",
    target_item="car",
    output_dir="my_output"  # 可選，自訂輸出目錄
)

# 處理結果
for file_path in saved_files:
    print(f"已儲存：{file_path}")
```

#### 3. 批次處理（範例）

```python
import os
from pathlib import Path

def batch_annotate(image_dir, target_item="objects"):
    """批次標註目錄中的所有圖片"""
    service = ImageAnnotationService()
    results = []
    
    for image_file in Path(image_dir).glob("*.jpg"):
        try:
            saved_files = service.annotate_image(
                image_path=str(image_file),
                target_item=target_item
            )
            results.append({
                "image": image_file.name,
                "status": "success",
                "saved_files": saved_files
            })
        except Exception as e:
            results.append({
                "image": image_file.name,
                "status": "error",
                "error": str(e)
            })
    
    return results

# 使用
results = batch_annotate("data/uploads", "person")
print(f"處理完成：{len(results)} 個圖片")
```

### 測試

```bash
# 執行服務測試
pytest tests/test_services.py::TestImageAnnotationService -v

# 執行範例腳本
python example_annotation.py

# 測試 API
curl -X POST http://localhost:8000/api/annotate-image \
  -F "file=@test_image.jpg" \
  -F "target_item=person"
```

---

## 故障排除

### 常見問題

#### 1. API Key 錯誤

**症狀**:
```
APIKeyError: Google API Key is not set
```

**解決方法**:
```bash
# 檢查 .env 檔案
cat .env | grep GOOGLE_API_KEY

# 設定 API Key
echo "GOOGLE_API_KEY=your-key" >> .env

# 或直接在程式中設定
service = ImageAnnotationService(api_key="your-key")
```

#### 2. 無法偵測物件

**症狀**:
- 回應為空
- 偵測數量為 0

**可能原因**:
1. 偵測目標描述不明確
2. 圖片品質不佳
3. 物件在圖片中不明顯

**解決方法**:
```python
# 1. 使用更通用的描述
target_item = "objects"  # 而不是 "specific_rare_object"

# 2. 確保圖片品質
from PIL import Image
img = Image.open("test.jpg")
print(f"圖片尺寸：{img.size}")
print(f"圖片模式：{img.mode}")

# 3. 嘗試不同的描述
for target in ["person", "objects", "items"]:
    result = service.detect_objects(image_path, target)
    print(f"{target}: {len(result)} 個物件")
```

#### 3. 圖片載入失敗

**症狀**:
```
ValidationError: Failed to load or process image
```

**解決方法**:
```bash
# 檢查檔案是否存在
ls -lh data/uploads/image.jpg

# 檢查檔案格式
file data/uploads/image.jpg

# 確認可以讀取
python -c "from PIL import Image; Image.open('data/uploads/image.jpg')"
```

#### 4. API 超時

**症狀**:
- 請求長時間無回應
- 超時錯誤

**解決方法**:
```python
# 1. 縮小圖片
from PIL import Image

img = Image.open("large_image.jpg")
img.thumbnail((1024, 1024))
img.save("resized_image.jpg")

# 2. 增加超時設定（如果可能）
# 3. 檢查網路連線
# 4. 稍後再試（API 可能繁忙）
```

#### 5. 座標不正確

**症狀**:
- 邊界框位置錯誤
- 座標超出範圍

**解決方法**:
```python
# 檢查座標系統
for obj in detected_objects:
    y_min, x_min, y_max, x_max = obj.box_2d
    print(f"原始座標: {obj.box_2d}")
    
    # 如果座標是 0-1000 範圍，轉換為像素
    width, height = image.size
    pixel_coords = [
        int(y_min * height / 1000),
        int(x_min * width / 1000),
        int(y_max * height / 1000),
        int(x_max * width / 1000)
    ]
    print(f"像素座標: {pixel_coords}")
```

### 偵錯技巧

#### 1. 啟用詳細日誌

```bash
# 設定日誌級別
export LOG_LEVEL=DEBUG

# 啟動服務
python run_api.py

# 查看日誌
tail -f logs/app.log | grep "ImageAnnotation"
```

#### 2. 檢查 API 回應

```python
import json

# 儲存 API 原始回應
response = service._call_gemini_api(image_path, target_item)
with open("api_response.json", "w") as f:
    json.dump(response, f, indent=2)

print("已儲存 API 回應到 api_response.json")
```

#### 3. 測試不同參數

```python
# 測試不同的偵測目標
targets = ["person", "car", "objects", "items"]
for target in targets:
    result = service.detect_objects("test.jpg", target)
    print(f"{target}: {len(result)} 個物件")

# 測試不同的圖片尺寸
sizes = [(512, 512), (1024, 1024), (2048, 2048)]
for size in sizes:
    service = ImageAnnotationService(max_image_size=size)
    # 測試...
```

### 效能優化

#### 1. 圖片預處理

```python
from PIL import Image

def optimize_image(image_path, max_size=(1024, 1024)):
    """優化圖片以加快處理速度"""
    img = Image.open(image_path)
    
    # 調整尺寸
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # 轉換為 RGB（如果需要）
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # 儲存優化後的圖片
    optimized_path = "optimized_" + os.path.basename(image_path)
    img.save(optimized_path, "JPEG", quality=85, optimize=True)
    
    return optimized_path
```

#### 2. 批次處理優化

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_annotate_batch(image_paths, target_item="objects"):
    """非同步批次標註"""
    service = ImageAnnotationService()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                service.annotate_image,
                path,
                target_item
            )
            for path in image_paths
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results
```

### 限制與注意事項

1. **API 配額**
   - Google API 有使用限制
   - 注意配額管理

2. **圖片大小**
   - 最大 5MB
   - 建議 1024x1024 以下

3. **偵測準確度**
   - 依賴 Gemini 模型能力
   - 複雜場景可能不準確

4. **處理時間**
   - 依網路速度和圖片大小
   - 通常 5-15 秒

5. **並發限制**
   - 避免同時大量請求
   - 建議加入延遲機制

---

## 相關資源

### 內部文檔
- [PROJECT_GUIDE.md](./PROJECT_GUIDE.md) - 專案完整指南
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 系統架構
- [TESTING.md](./TESTING.md) - 測試指南

### 外部資源
- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API 文檔](https://ai.google.dev/docs)
- [Pillow 文檔](https://pillow.readthedocs.io/)

### 範例程式碼
- `example_annotation.py` - 完整使用範例
- `test_image_annotation.py` - 測試腳本

---

## 授權

MIT License

---

**🎉 開始使用智慧圖片標註功能！**

**版本**: 1.0  
**更新日期**: 2025-11-14  
**狀態**: ✅ 穩定

---

*享受 AI 驅動的圖片標註體驗！* 🚀
