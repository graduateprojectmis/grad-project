#!/bin/bash

# 前端啟動腳本

echo "🎨 啟動 AirPods Q&A 前端服務"
echo "================================"

# 檢查 Python 環境
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python 未安裝或不在 PATH 中"
    exit 1
fi

# 使用 python3 或 python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "📂 進入 web/frontend 目錄..."
cd web/frontend

echo ""
echo "🚀 啟動前端服務..."
echo "前端將在 http://localhost:8080 運行"
echo ""
echo "💡 提示："
echo "1. 確保後端 API 已經啟動（http://localhost:8000）"
echo "2. 開啟瀏覽器訪問：http://localhost:8080"
echo ""
echo "按 Ctrl+C 停止服務"
echo ""

# 啟動前端
$PYTHON_CMD -m http.server 8080
