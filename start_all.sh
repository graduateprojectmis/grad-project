#!/bin/bash

# AirPods Q&A 系統 - 一鍵啟動腳本 (macOS)

echo ""
echo "===================================="
echo " 🎧 AirPods Q&A 系統一鍵啟動"
echo "===================================="
echo ""

# 檢查是否為 macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  警告：此腳本專為 macOS 設計"
fi

# 檢查 Python 環境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安裝"
    echo "請使用 Homebrew 安裝：brew install python3"
    exit 1
fi

PYTHON_CMD="python3"
PIP_CMD="pip3"

echo "🔍 檢查環境..."
$PYTHON_CMD --version

# 檢查是否在正確的目錄
if [ ! -f "web/backend/api.py" ]; then
    echo "❌ 請在專案根目錄執行此腳本"
    exit 1
fi

# 檢查 .env 文件
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  警告：.env 檔案不存在"
    echo ""
    echo "請使用以下方式之一設定 API Key："
    echo ""
    echo "方式 1 - 使用設置腳本（推薦）："
    echo "  ./setup_env.sh"
    echo ""
    echo "方式 2 - 手動創建 .env 檔案："
    echo "  echo 'OPENAI_API_KEY=sk-your-api-key' > .env"
    echo "  chmod 600 .env"
    echo ""
    read -p "是否繼續？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ 找到 .env 檔案"
fi

echo ""
echo "📦 安裝依賴套件..."
$PIP_CMD install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ 依賴安裝失敗"
    exit 1
fi

echo ""
echo "🔧 檢查 ChromaDB 資料庫..."
if [ ! -d "web/backend/chroma_db" ]; then
    echo "⚠️  ChromaDB 資料庫不存在"
    echo "🔄 正在初始化資料庫..."
    $PYTHON_CMD init_chromadb.py
    if [ $? -ne 0 ]; then
        echo "❌ 初始化失敗"
        exit 1
    fi
else
    echo "✅ ChromaDB 資料庫已存在"
fi

echo ""
echo "===================================="
echo " 🚀 啟動服務"
echo "===================================="
echo ""
echo "📋 服務資訊："
echo "  • 後端 API: http://localhost:8000"
echo "  • API 文檔: http://localhost:8000/api/docs"
echo "  • 前端介面: http://localhost:8080"
echo ""
echo "💡 提示："
echo "  • 兩個服務將在背景執行"
echo "  • 按 Ctrl+C 停止所有服務"
echo "  • 日誌儲存在 logs/ 目錄"
echo ""

# 建立 logs 目錄
mkdir -p logs

# 啟動後端
echo "🔧 啟動後端 API..."
cd web/backend
$PYTHON_CMD api.py > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

# 等待後端啟動
echo "⏳ 等待後端啟動..."
sleep 3

# 檢查後端是否正常啟動
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ 後端啟動失敗"
    echo "📄 查看日誌：tail -f logs/backend.log"
    exit 1
fi

echo "✅ 後端已啟動 (PID: $BACKEND_PID)"

# 啟動前端
echo "🎨 啟動前端服務..."
cd web/frontend
$PYTHON_CMD -m http.server 8080 > ../../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

# 等待前端啟動
sleep 2

# 檢查前端是否正常啟動
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ 前端啟動失敗"
    echo "📄 查看日誌：tail -f logs/frontend.log"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ 前端已啟動 (PID: $FRONTEND_PID)"

echo ""
echo "===================================="
echo " ✅ 所有服務啟動完成！"
echo "===================================="
echo ""
echo "🌐 在瀏覽器開啟：http://localhost:8080"
echo ""
echo "📝 即時日誌："
echo "  tail -f logs/backend.log   # 後端日誌"
echo "  tail -f logs/frontend.log  # 前端日誌"
echo ""
echo "🛑 停止服務："
echo "  ./stop_all.sh              # 或按 Ctrl+C"
echo ""

# 保存 PID 到檔案
echo $BACKEND_PID > logs/backend.pid
echo $FRONTEND_PID > logs/frontend.pid

# 捕捉 Ctrl+C 信號
cleanup() {
    echo ""
    echo "🛑 正在停止服務..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    rm -f logs/backend.pid logs/frontend.pid
    echo "✅ 所有服務已停止"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 自動在瀏覽器開啟（macOS）
if command -v open &> /dev/null; then
    sleep 1
    open http://localhost:8080
fi

# 保持腳本運行
wait

