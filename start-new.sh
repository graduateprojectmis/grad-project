#!/bin/bash
# 啟動腳本（重構版）

echo "🚀 啟動 AirPods Q&A 系統（重構版 v2.0）"
echo "================================================"

# 檢查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安裝！"
    echo "請先安裝 Python 3.8 或更高版本"
    exit 1
fi

echo "✅ Python 版本："
python3 --version

# 檢查 .env 檔案
if [ ! -f .env ]; then
    echo "⚠️  .env 檔案不存在，建立範例檔案..."
    cp .env.example .env
    echo "📝 請編輯 .env 檔案並設定您的 API Key"
    echo "   OPENAI_API_KEY=sk-your-api-key-here"
    exit 1
fi

# 檢查依賴
echo "📦 檢查依賴套件..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  依賴套件未安裝，正在安裝..."
    pip3 install -r requirements-new.txt
fi

# 檢查資料庫
if [ ! -d "data/chroma_db" ]; then
    echo "⚠️  ChromaDB 尚未初始化"
    echo "📊 正在初始化資料..."
    python3 init_data.py

fi

# 啟動服務
echo "🚀 啟動 API 伺服器..."
echo "================================================"
echo "📍 API 服務: http://localhost:8000"
echo "📚 API 文檔: http://localhost:8000/api/docs"
echo "================================================"
echo ""

python3 run_api.py
