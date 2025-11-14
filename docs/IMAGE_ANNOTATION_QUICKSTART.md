# 圖片標記功能快速開始

## 🚀 快速開始（5 分鐘設定）

### 1️⃣ 設定 Google API Key

編輯 `.env` 檔案，新增：

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

> **如何取得 Google API Key？**
> 1. 訪問 [Google AI Studio](https://makersuite.google.com/app/apikey)
> 2. 登入 Google 帳號
> 3. 點擊「Create API Key」
> 4. 複製 API Key 並貼到 `.env` 檔案

### 2️⃣ 啟動服務

```bash
# 啟動後端
python run_api.py

# 另開終端，啟動前端
cd frontend-react
npm run dev
```

### 3️⃣ 測試功能

1. 開啟瀏覽器訪問 `http://localhost:5173`
2. 點擊 📎 按鈕選擇圖片
3. ✅ **勾選「啟用圖片標記」**
4. 輸入偵測目標（例如：`person`, `car`, `objects`）
5. 點擊「發送」

### 4️⃣ 查看結果

系統會顯示：
- 偵測到的物件數量
- 每個物件的標籤和座標
- 標註圖片儲存位置

標註圖片位於：
```
data/output/Annotated_Image/
```

## 📸 使用範例

### 範例 1：偵測人物

```
1. 上傳包含人物的照片
2. 啟用圖片標記
3. 偵測目標輸入：person
4. 發送
```

**結果：**
```
✅ 成功偵測並標註 2 個物件
偵測到的物件：
1. person (座標: [150, 200, 400, 500])
2. person (座標: [450, 180, 700, 520])

已儲存 2 個標註圖片
```

### 範例 2：偵測多種物件

```
1. 上傳街景照片
2. 啟用圖片標記
3. 偵測目標輸入：objects
4. 發送
```

**結果：**
```
✅ 成功偵測並標註 5 個物件
偵測到的物件：
1. car (座標: [100, 150, 300, 400])
2. person (座標: [320, 180, 450, 520])
3. traffic light (座標: [500, 50, 550, 200])
4. building (座標: [0, 0, 800, 300])
5. tree (座標: [750, 100, 900, 600])
```

## 🎯 常用偵測目標

| 偵測目標 | 說明 | 範例 |
|---------|------|------|
| `person` | 人物 | 偵測照片中的人 |
| `car` | 汽車 | 偵測街景中的車輛 |
| `dog` | 狗 | 偵測寵物照片中的狗 |
| `cat` | 貓 | 偵測寵物照片中的貓 |
| `chair` | 椅子 | 偵測室內傢俱 |
| `table` | 桌子 | 偵測室內傢俱 |
| `laptop` | 筆記型電腦 | 偵測桌面物品 |
| `phone` | 手機 | 偵測電子產品 |
| `tree` | 樹木 | 偵測自然景物 |
| `objects` | 一般物件 | 偵測所有物件（預設）|

## 💡 使用技巧

### 技巧 1：提高準確度

- ✅ 使用高解析度圖片
- ✅ 確保物件清晰可見
- ✅ 使用具體的偵測目標描述

### 技巧 2：批次偵測

如果想偵測多種物件，可以：

1. 使用通用描述：`objects`, `items`, `things`
2. 或具體列舉：`person and car`, `furniture`

### 技巧 3：檢視標註結果

標註圖片會自動儲存，可以在以下位置查看：

```bash
# macOS/Linux
open data/output/Annotated_Image/

# Windows
explorer data\output\Annotated_Image\
```

## ⚠️ 注意事項

1. **需要網路連線**：標註功能需要呼叫 Google API
2. **處理時間**：視圖片大小和複雜度，可能需要 5-15 秒
3. **API 額度**：Google API 有使用限制，請注意配額
4. **圖片大小**：建議使用小於 5MB 的圖片

## 🔧 故障排除

### 問題：無法標註圖片

**檢查清單：**
```bash
# 1. 確認 Google API Key 是否設定
cat .env | grep GOOGLE_API_KEY

# 2. 測試 API 連接
python test_image_annotation.py

# 3. 檢查後端日誌
tail -f logs/app.log
```

### 問題：偵測不到物件

**解決方法：**
1. 使用更通用的描述（如 `objects`）
2. 確保圖片品質良好
3. 嘗試不同的偵測目標

### 問題：處理速度慢

**原因：**
- 圖片太大
- 網路連線慢
- API 服務繁忙

**解決方法：**
1. 壓縮圖片
2. 檢查網路連線
3. 稍後再試

## 📚 相關文件

- [完整功能說明](./IMAGE_ANNOTATION_FEATURE.md)
- [實現細節](./IMAGE_ANNOTATION_IMPLEMENTATION.md)
- [API 文件](../README.md)

## 🎉 開始使用

現在你已經準備好使用圖片標記功能了！

1. 確保 `.env` 已設定 Google API Key
2. 啟動服務
3. 上傳圖片
4. 啟用標記開關
5. 享受 AI 圖片標註！

祝你使用愉快！ 🚀
