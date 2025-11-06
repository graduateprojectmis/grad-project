# 🎨 AirPods Q&A - React Frontend

> 基於 React 18 + Vite 的現代化前端應用

## ✨ 特色

- ⚡ **Vite** - 極速開發和構建
- ⚛️ **React 18** - 最新的 React 特性
- 🎯 **組件化設計** - 清晰的組件架構
- 🎨 **現代化 UI** - 優雅的使用者介面
- 📱 **響應式設計** - 完美支援行動裝置
- 🔄 **狀態管理** - React Hooks
- 🌐 **API 整合** - Axios + RESTful API
- 🎭 **圖標系統** - Lucide React Icons

## 📁 專案結構

```
frontend-react/
├── public/                  # 靜態資源
├── src/
│   ├── components/         # React 元件
│   │   ├── Header.jsx          # 頁首元件
│   │   ├── ChatContainer.jsx   # 聊天容器
│   │   ├── Message.jsx         # 訊息元件
│   │   ├── WelcomeSection.jsx  # 歡迎區域
│   │   ├── InputArea.jsx       # 輸入區域
│   │   └── ApiKeyModal.jsx     # API Key 設定彈窗
│   ├── services/           # API 服務
│   │   └── api.js              # API 請求封裝
│   ├── App.jsx             # 主應用元件
│   ├── App.css             # 主樣式
│   ├── main.jsx            # 應用入口
│   └── index.css           # 全域樣式
├── index.html              # HTML 模板
├── vite.config.js          # Vite 配置
├── package.json            # 依賴管理
├── .env.example            # 環境變數範例
└── README.md               # 本文件
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
cd frontend-react
npm install
```

### 2. 設定環境變數

```bash
# 複製環境變數範例
cp .env.example .env

# 編輯 .env（如果需要）
# VITE_API_URL=http://localhost:8000
```

### 3. 啟動開發伺服器

```bash
npm run dev
```

應用將在 http://localhost:3000 啟動

### 4. 構建生產版本

```bash
npm run build
```

構建文件將輸出到 `dist/` 目錄

### 5. 預覽生產構建

```bash
npm run preview
```

## 📦 技術棧

### 核心依賴

| 套件 | 版本 | 說明 |
|------|------|------|
| react | ^18.3.1 | React 核心庫 |
| react-dom | ^18.3.1 | React DOM 渲染 |
| axios | ^1.7.9 | HTTP 請求庫 |
| lucide-react | ^0.460.0 | React 圖標庫 |

### 開發依賴

| 套件 | 版本 | 說明 |
|------|------|------|
| vite | ^6.0.1 | 建置工具 |
| @vitejs/plugin-react | ^4.3.4 | Vite React 插件 |
| eslint | ^9.15.0 | 程式碼檢查 |

## 🧩 元件說明

### Header
- 顯示應用標題
- API 連接狀態指示器
- 主頁按鈕和設定按鈕

### ChatContainer
- 訊息顯示容器
- 自動滾動到最新訊息
- 包含歡迎區域

### Message
- 單一訊息元件
- 支援使用者和機器人訊息
- 支援錯誤訊息樣式

### WelcomeSection
- 歡迎畫面
- 範例問題按鈕
- 引導使用者開始對話

### InputArea
- 問題輸入框
- 圖片上傳功能
- 發送按鈕
- 載入狀態處理

### ApiKeyModal
- API Key 管理彈窗
- 檢查 Key 狀態
- 儲存/刪除 API Key

## 🔌 API 整合

### API 服務層 (`services/api.js`)

所有 API 請求都通過 Axios 實例統一管理：

```javascript
// 檢查 API 健康狀態
await checkApiHealth()

// 發送問題
await askQuestion(question, topK)

// API Key 管理
await checkApiKeyStatus()
await saveApiKey(apiKey)
await deleteApiKey()

// 圖片上傳
await uploadImage(file)
```

### 代理配置

開發環境使用 Vite 代理避免 CORS 問題：

```javascript
// vite.config.js
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    }
  }
}
```

## 🎨 樣式系統

### CSS 變數

```css
:root {
  --primary: #D97757;
  --primary-dark: #C86A47;
  --bg: #F5F3EF;
  --surface: #FFFFFF;
  --text: #2C2416;
  --text-light: #5C5448;
  --border: #E8E3DB;
  --user-bg: #E8E8E8;
  --bot-bg: #F9F7F4;
}
```

### 組件樣式

每個組件都有獨立的 CSS 文件，遵循 CSS Modules 模式。

## 📱 響應式設計

支援以下斷點：

- **Desktop**: > 768px
- **Mobile**: ≤ 768px

關鍵調整：
- 隱藏次要文字標籤
- 簡化網格佈局
- 調整字體大小
- 優化觸控目標

## 🔧 開發指南

### 新增元件

1. 在 `src/components/` 創建 `.jsx` 文件
2. 創建對應的 `.css` 文件
3. 在 `App.jsx` 中引入並使用

範例：

```jsx
// src/components/NewComponent.jsx
import './NewComponent.css'

function NewComponent({ prop1, prop2 }) {
  return (
    <div className="new-component">
      {/* 元件內容 */}
    </div>
  )
}

export default NewComponent
```

### 新增 API 方法

在 `src/services/api.js` 中添加新方法：

```javascript
export const newApiMethod = async (param) => {
  const response = await api.post('/api/endpoint', { param })
  return response.data
}
```

## 🚢 部署

### 構建

```bash
npm run build
```

### 部署到靜態伺服器

將 `dist/` 目錄部署到任何靜態檔案伺服器：

- **Netlify**: 拖放 `dist/` 資料夾
- **Vercel**: 連接 Git 倉庫
- **GitHub Pages**: 使用 `gh-pages` 分支
- **Nginx**: 配置指向 `dist/` 目錄

### 環境變數

生產環境需要設定：

```bash
VITE_API_URL=https://your-api-domain.com
```

## 🔍 常見問題

### 1. CORS 錯誤

**問題**: 開發時遇到 CORS 錯誤

**解決**: Vite 代理已配置，確保後端運行在 `localhost:8000`

### 2. API 連接失敗

**問題**: 前端無法連接後端

**解決**: 
- 檢查後端是否運行
- 檢查 `.env` 中的 `VITE_API_URL`
- 檢查網路連接

### 3. 圖片上傳失敗

**問題**: 無法上傳圖片

**解決**:
- 檢查圖片大小 (< 5MB)
- 檢查圖片格式 (JPG, PNG, GIF, WEBP)
- 檢查後端上傳端點

## 📚 相關文檔

- [React 文檔](https://react.dev/)
- [Vite 文檔](https://vitejs.dev/)
- [Axios 文檔](https://axios-http.com/)
- [Lucide Icons](https://lucide.dev/)

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

MIT License

---

**專案版本**: 2.0  
**最後更新**: 2025-11-07  
**維護者**: 專案團隊
