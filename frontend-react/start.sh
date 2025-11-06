#!/bin/bash
# React Frontend 開發啟動腳本

set -e

echo "======================================"
echo "🎨 啟動 React Frontend"
echo "======================================"
echo ""

cd "$(dirname "$0")"

# 檢查 node_modules
if [ ! -d "node_modules" ]; then
    echo "📦 安裝依賴中..."
    npm install
    echo ""
fi

# 檢查 .env 文件
if [ ! -f ".env" ]; then
    echo "📝 創建 .env 文件..."
    cp .env.example .env
    echo "✅ .env 文件已創建"
    echo ""
fi

echo "🚀 啟動開發伺服器..."
echo "📍 前端: http://localhost:3000"
echo "📍 後端: http://localhost:8000"
echo ""
echo "按 Ctrl+C 停止伺服器"
echo "======================================"
echo ""

npm run dev
