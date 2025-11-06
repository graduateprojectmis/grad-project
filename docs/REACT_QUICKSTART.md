# ⚛️ React 前端快速啟動指南

## 🚀 三種啟動方式

### 方式 1: 一鍵啟動（推薦）

啟動整個系統（後端 + React 前端）：

```bash
./start-react.sh
```

這會自動：
- ✅ 啟動後端 API (http://localhost:8000)
- ✅ 啟動 React 前端 (http://localhost:3000)
- ✅ 自動安裝依賴（首次運行）
- ✅ 自動開啟瀏覽器

### 方式 2: 分別啟動

#### 啟動後端
```bash
python run_api.py
```

#### 啟動 React 前端
```bash
cd frontend-react
./start.sh
```

### 方式 3: 手動啟動

```bash
# 終端 1 - 後端
python run_api.py

# 終端 2 - 前端
cd frontend-react
npm install    # 首次運行
npm run dev
```

---

## 📦 首次安裝

### 1. 安裝 Node.js

```bash
# macOS (使用 Homebrew)
brew install node

# 或下載安裝器
# https://nodejs.org/
```

### 2. 安裝前端依賴

```bash
cd frontend-react
npm install
```

### 3. 配置環境變數（可選）

```bash
cd frontend-react
cp .env.example .env
# 編輯 .env 如需自訂 API URL
```

---

## 🎯 訪問應用

啟動成功後：

- **前端應用**: http://localhost:3000
- **後端 API**: http://localhost:8000
- **API 文檔**: http://localhost:8000/docs

---

## 🔧 常用命令

### 開發

```bash
cd frontend-react

# 啟動開發伺服器
npm run dev

# 程式碼檢查
npm run lint

# 構建生產版本
npm run build

# 預覽生產構建
npm run preview
```

### 清理

```bash
# 清理依賴
rm -rf node_modules
npm install

# 清理構建
rm -rf dist
```

---

## ❓ 常見問題

### 1. Port 3000 已被佔用

**解決方法**:

```bash
# 方法 1: 修改 vite.config.js
server: {
  port: 3001,  # 改成其他端口
}

# 方法 2: 結束佔用的進程
lsof -ti:3000 | xargs kill
```

### 2. 無法連接後端

**檢查清單**:
- ✅ 後端是否運行？`curl http://localhost:8000/api/health`
- ✅ 端口是否正確？檢查 `.env` 中的 `VITE_API_URL`
- ✅ 防火牆設定？

### 3. 依賴安裝失敗

```bash
# 清除 npm 緩存
npm cache clean --force

# 刪除 package-lock.json 和 node_modules
rm -rf node_modules package-lock.json

# 重新安裝
npm install
```

### 4. HMR 不工作

```bash
# 重啟開發伺服器
# Ctrl+C 停止
npm run dev
```

---

## 📱 開發技巧

### 1. 開啟 React DevTools

安裝瀏覽器擴充：
- [Chrome](https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi)
- [Firefox](https://addons.mozilla.org/en-US/firefox/addon/react-devtools/)

### 2. 查看網路請求

在瀏覽器開發者工具中：
- Network 標籤 → 查看 API 請求
- Console 標籤 → 查看錯誤訊息

### 3. 調試組件

在程式碼中添加：

```jsx
console.log('State:', someState)
```

---

## 🎨 自訂配置

### 修改 API URL

編輯 `frontend-react/.env`:

```bash
VITE_API_URL=http://your-api-url:8000
```

### 修改端口

編輯 `frontend-react/vite.config.js`:

```javascript
server: {
  port: 3001,  // 改成你想要的端口
}
```

### 修改主題色

編輯 `frontend-react/src/index.css`:

```css
:root {
  --primary: #D97757;  /* 主色調 */
  --bg: #F5F3EF;       /* 背景色 */
  /* ... */
}
```

---

## 📚 專案結構

```
frontend-react/
├── src/
│   ├── components/      # React 元件
│   ├── services/        # API 服務
│   ├── App.jsx          # 主應用
│   └── main.jsx         # 入口
├── public/              # 靜態資源
├── package.json         # 依賴配置
├── vite.config.js       # Vite 配置
└── .env                 # 環境變數
```

---

## 🚢 部署

### 構建

```bash
cd frontend-react
npm run build
```

### 部署到 Netlify

```bash
# 1. 安裝 Netlify CLI
npm install -g netlify-cli

# 2. 登入
netlify login

# 3. 部署
netlify deploy --prod
```

### 部署到 Vercel

```bash
# 1. 安裝 Vercel CLI
npm install -g vercel

# 2. 部署
vercel
```

---

## 💡 下一步

1. **設定 API Key** - 首次使用時設定 OpenAI API Key
2. **試用功能** - 點擊範例問題開始對話
3. **探索代碼** - 查看 React 元件結構
4. **自訂修改** - 根據需求調整樣式和功能

---

## 📞 獲取幫助

- **文檔**: [frontend-react/README.md](frontend-react/README.md)
- **報告**: [REACT_REFACTORING_REPORT.md](REACT_REFACTORING_REPORT.md)
- **後端**: [README-NEW.md](docs/README-NEW.md)

---

**快速開始，享受 React 的開發體驗！** ⚛️✨
