# 圖像標註服務 (Image Annotation Service)

## 概述

`ImageAnnotationService` 是一個使用 Google Gemini API 進行物件偵測和邊界框標註的服務模組。它能夠：

- 偵測圖像中的特定物件
- 生成帶有邊界框和標籤的標註圖像
- 提供偵測摘要資訊

## 功能特點

- ✅ 基於 Google Gemini 2.0 Flash 模型進行物件偵測
- ✅ 自動調整圖像大小以符合 API 限制
- ✅ 繪製清晰的邊界框和標籤
- ✅ 完整的錯誤處理和日誌記錄
- ✅ 遵循專案的服務層架構
- ✅ 完整的單元測試覆蓋

## 安裝依賴

確保已安裝以下套件：

```bash
pip install google-genai pillow
```

## 環境設定

在 `.env` 檔案中設定 Google API Key：

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## 使用方式

### 1. 基本使用

```python
from app.services import ImageAnnotationService

# 初始化服務
service = ImageAnnotationService()

# 標註圖像並儲存
saved_files = service.annotate_image(
    image_path="data/uploads/screenshot.png",
    target_item="play button"
)

print(f"已儲存 {len(saved_files)} 個標註圖像")
```

### 2. 僅獲取偵測資訊（不儲存圖像）

```python
# 獲取偵測摘要
summary = service.get_detection_summary(
    image_path="data/uploads/screenshot.png",
    target_item="play button"
)

print(f"偵測到 {summary['total_detected']} 個物件")
for obj in summary['objects']:
    print(f"- {obj['label']}: {obj['box_2d']}")
```

### 3. 僅偵測物件

```python
# 只執行物件偵測
detected_objects = service.detect_objects(
    image_path="data/uploads/screenshot.png",
    target_item="play button"
)

for obj in detected_objects:
    print(f"{obj.label}: {obj.box_2d}")
```

### 4. 自訂參數

```python
# 使用自訂參數初始化
service = ImageAnnotationService(
    api_key="your_custom_key",
    model="gemini-2.0-flash-exp",
    max_image_size=(2048, 2048)
)

# 指定輸出目錄
saved_files = service.annotate_image(
    image_path="test.png",
    target_item="button",
    output_dir="custom_output"
)
```

## API 參考

### ImageAnnotationService 類別

#### `__init__(api_key=None, model="gemini-2.0-flash-exp", max_image_size=(1024, 1024))`

初始化圖像標註服務。

**參數：**
- `api_key` (str, optional): Google API Key，預設從環境變數讀取
- `model` (str, optional): Gemini 模型名稱
- `max_image_size` (tuple, optional): 圖像最大尺寸 (寬, 高)

#### `detect_objects(image_path, target_item)`

偵測圖像中的物件。

**參數：**
- `image_path` (str): 圖像檔案路徑
- `target_item` (str): 目標物件描述

**回傳：**
- `List[DetectedObject]`: 偵測到的物件列表

#### `annotate_image(image_path, target_item, output_dir=None)`

對圖像進行標註並儲存結果。

**參數：**
- `image_path` (str): 輸入圖像路徑
- `target_item` (str): 目標物件描述
- `output_dir` (str, optional): 輸出目錄路徑

**回傳：**
- `List[str]`: 儲存的檔案路徑列表

#### `get_detection_summary(image_path, target_item)`

獲取偵測摘要資訊。

**參數：**
- `image_path` (str): 圖像檔案路徑
- `target_item` (str): 目標物件描述

**回傳：**
- `Dict`: 包含偵測資訊的字典

### DetectedObject 類別

表示偵測到的物件。

**屬性：**
- `box_2d` (List[int]): 2D 邊界框座標 [y_min, x_min, y_max, x_max]，範圍 0-1000
- `label` (str): 物件標籤

**方法：**
- `to_dict()`: 轉換為字典格式

## 範例腳本

執行範例腳本以查看完整功能：

```bash
python example_annotation.py
```

## 輸出格式

### 標註圖像

- 檔案名稱格式：`{label}_{index}_detected.png`
- 例如：`play_button_0_detected.png`

### 偵測摘要

```python
{
    "image_path": "data/uploads/screenshot.png",
    "target_item": "play button",
    "total_detected": 2,
    "objects": [
        {
            "box_2d": [100, 200, 300, 400],
            "label": "play_button"
        },
        {
            "box_2d": [500, 600, 700, 800],
            "label": "pause_button"
        }
    ]
}
```

## 座標系統

邊界框座標使用 Gemini API 的標準格式：

- `[y_min, x_min, y_max, x_max]`
- 座標範圍：0-1000（正規化座標）
- 會自動轉換為實際圖像像素座標

## 錯誤處理

服務使用專案的統一例外處理機制：

- `APIKeyError`: API Key 未設定或無效
- `ValidationError`: 圖像載入失敗或 API 回應格式錯誤
- 所有錯誤都會記錄到日誌中

## 日誌記錄

服務使用專案的統一日誌系統，記錄：

- 初始化資訊
- 偵測過程
- 錯誤和警告
- 成功操作

查看日誌：

```bash
tail -f logs/app.log
```

## 測試

執行單元測試：

```bash
# 執行所有服務測試
pytest tests/test_services.py::TestImageAnnotationService -v

# 執行特定測試
pytest tests/test_services.py::TestImageAnnotationService::test_detect_objects -v
```

## 整合到 API

如需將此服務整合到 FastAPI 端點，參考以下範例：

```python
from fastapi import APIRouter, UploadFile, File
from app.services import ImageAnnotationService

router = APIRouter()

@router.post("/annotate")
async def annotate_image(
    file: UploadFile = File(...),
    target_item: str = "button"
):
    # 儲存上傳的檔案
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # 執行標註
    service = ImageAnnotationService()
    saved_files = service.annotate_image(file_path, target_item)
    
    return {
        "message": "標註完成",
        "files": saved_files
    }
```

## 效能考量

- 圖像會自動調整為 1024x1024 以下
- 每次 API 呼叫處理一張圖像
- 建議批次處理大量圖像時加入適當的延遲

## 限制

- 僅支援 Gemini 2.0 Flash 及以上版本
- 圖像大小限制：最大 1024x1024（可調整）
- API 呼叫次數受 Google Cloud 配額限制

## 進階使用

### 自訂邊界框樣式

修改 `_draw_bounding_box` 方法中的參數：

```python
# 在服務類別中
composite = self._draw_bounding_box(
    image=image,
    detected_object=obj,
    box_color=(0, 255, 0, 255),  # 綠色邊界框
    box_width=3  # 細線
)
```

## 疑難排解

### 問題：API Key 錯誤

**解決方案：**
1. 確認 `.env` 檔案中有設定 `GOOGLE_API_KEY`
2. 檢查 API Key 是否有效
3. 確認已啟用 Gemini API

### 問題：圖像無法載入

**解決方案：**
1. 確認圖像路徑正確
2. 確認圖像格式支援（PNG, JPG, JPEG）
3. 檢查圖像檔案是否損壞

### 問題：偵測結果不準確

**解決方案：**
1. 使用更具體的目標物件描述
2. 確保圖像品質良好
3. 嘗試不同的 Gemini 模型

## 貢獻

歡迎提交 Issue 或 Pull Request 來改進此服務。

## 授權

遵循專案主授權協議。
