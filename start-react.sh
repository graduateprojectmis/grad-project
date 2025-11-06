#!/bin/bash
# 一鍵啟動整個系統（後端 + React 前端）

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "================================================"
echo "🚀 啟動 AirPods Q&A 系統 (React 版本)"
echo "================================================"
echo ""

# 檢查 Python 環境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安裝"
    exit 1
fi

# 檢查 Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安裝"
    exit 1
fi

# 啟動後端 API
echo "📡 啟動後端 API..."
cd "$PROJECT_DIR"
python run_api.py > logs/api.log 2>&1 &
API_PID=$!
echo "✅ 後端 API 已啟動 (PID: $API_PID)"
echo "📍 http://localhost:8000"
echo ""

# 等待後端啟動
echo "⏳ 等待後端就緒..."
sleep 3

# 啟動 React 前端
echo "🎨 啟動 React 前端..."
cd "$PROJECT_DIR/frontend-react"

if [ ! -d "node_modules" ]; then
    echo "📦 首次運行，安裝依賴..."
    npm install
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
fi

npm run dev &
FRONTEND_PID=$!
echo "✅ React 前端已啟動 (PID: $FRONTEND_PID)"
echo "📍 http://localhost:3000"
echo ""

echo "================================================"
echo "✅ 系統已啟動完成！"
echo "================================================"
echo ""
echo "📍 前端: http://localhost:3000"
echo "📍 後端: http://localhost:8000"
echo "📍 API 文檔: http://localhost:8000/docs"
echo ""
echo "💡 提示:"
echo "  - 首次使用請先設定 OpenAI API Key"
echo "  - 按 Ctrl+C 停止所有服務"
echo ""
echo "================================================"

# 等待中斷信號
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; echo ''; echo '👋 系統已停止'; exit" INT TERM

# 保持腳本運行
wait
