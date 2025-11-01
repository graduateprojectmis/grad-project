#!/bin/bash

# AirPods Q&A 系統 - 停止腳本 (macOS)

echo ""
echo "===================================="
echo " 🛑 停止 AirPods Q&A 系統"
echo "===================================="
echo ""

STOPPED=0

# 從 PID 檔案停止
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo "🔧 停止後端服務 (PID: $BACKEND_PID)"
        kill $BACKEND_PID 2>/dev/null
        STOPPED=1
    fi
    rm -f logs/backend.pid
fi

if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "🎨 停止前端服務 (PID: $FRONTEND_PID)"
        kill $FRONTEND_PID 2>/dev/null
        STOPPED=1
    fi
    rm -f logs/frontend.pid
fi

# 等待進程結束
if [ $STOPPED -eq 1 ]; then
    sleep 1
fi

# 額外清理：查找並停止在 port 8000 和 8080 上的進程
echo "🔍 檢查殘留進程..."

# 停止 port 8000 (後端)
PID_8000=$(lsof -ti:8000 2>/dev/null)
if [ ! -z "$PID_8000" ]; then
    echo "🔧 清理後端進程 (PID: $PID_8000)"
    kill -9 $PID_8000 2>/dev/null
    STOPPED=1
fi

# 停止 port 8080 (前端)
PID_8080=$(lsof -ti:8080 2>/dev/null)
if [ ! -z "$PID_8080" ]; then
    echo "🎨 清理前端進程 (PID: $PID_8080)"
    kill -9 $PID_8080 2>/dev/null
    STOPPED=1
fi

echo ""
if [ $STOPPED -eq 1 ]; then
    echo "✅ 所有服務已停止"
else
    echo "ℹ️  沒有運行中的服務"
fi
echo ""

