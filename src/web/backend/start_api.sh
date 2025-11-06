#!/bin/bash

# AirPods Q&A 系統啟動腳本

echo "🎧 AirPods Q&A 系統啟動腳本"
echo "================================"

# 檢查 Python 環境
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python 未安裝或不在 PATH 中"
    exit 1
fi

# 使用 python3 或 python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

# 檢查是否在正確的目錄
if [ ! -f "api.py" ]; then
    echo "❌ 請在 web/backend 目錄中執行此腳本"
    exit 1
fi

echo "📦 安裝依賴套件..."
$PIP_CMD install -r ../../requirements.txt

echo ""
echo "🔧 檢查 ChromaDB 資料..."
if [ ! -d "chroma_db" ] || [ ! -f "../../output/json/text_embedding_openai.json" ]; then
    echo "⚠️  警告：ChromaDB 資料不存在"
    echo "請先執行以下命令初始化資料："
    echo "cd ../.. && python tools/ChromaDB.py"
    echo ""
    read -p "是否現在初始化資料？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 初始化 ChromaDB 資料..."
        cd ../.. && $PYTHON_CMD tools/ChromaDB.py && cd web/backend
    else
        echo "⚠️  跳過資料初始化，API 可能無法正常運作"
    fi
fi

echo ""
echo "🚀 啟動 API 服務..."
echo "API 將在 http://localhost:8000 運行"
echo "API 文檔：http://localhost:8000/api/docs"
echo ""
echo "💡 提示："
echo "1. 開啟另一個終端機"
echo "2. 進入專案根目錄執行：./start_frontend.sh"
echo "3. 或進入 frontend 目錄執行：python -m http.server 8080"
echo "4. 開啟瀏覽器訪問：http://localhost:8080"
echo ""
echo "按 Ctrl+C 停止服務"
echo ""

# 啟動 API
$PYTHON_CMD api.py
