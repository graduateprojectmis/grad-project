#!/bin/bash

# AirPods Q&A 前端啟動腳本

echo "🎧 AirPods Q&A 前端啟動腳本"
echo "================================"

# 檢查是否在正確的目錄
if [ ! -f "index.html" ]; then
    echo "❌ 請在 grad-project-frontend 目錄中執行此腳本"
    exit 1
fi

echo "🌐 啟動前端服務..."
echo "前端將在 http://localhost:8080 運行"
echo ""
echo "💡 提示："
echo "1. 確保後端 API 正在運行 (http://localhost:8000)"
echo "2. 如果後端未運行，請先執行："
echo "   cd ../grad-project-main"
echo "   ./start_api.sh"
echo ""
echo "按 Ctrl+C 停止服務"
echo ""

# 檢查 Python 是否可用
if command -v python3 &> /dev/null; then
    python3 -m http.server 8080
elif command -v python &> /dev/null; then
    python -m http.server 8080
else
    echo "❌ Python 未安裝或不在 PATH 中"
    echo "請手動啟動 HTTP 伺服器"
    exit 1
fi
