#!/bin/bash

# .env 檔案設置腳本

echo ""
echo "===================================="
echo " 🔐 設定 API Key"
echo "===================================="
echo ""

ENV_FILE=".env"

# 檢查 .env 是否已存在
if [ -f "$ENV_FILE" ]; then
    echo "⚠️  .env 檔案已存在"
    echo ""
    read -p "是否要覆蓋現有設定？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消操作"
        exit 0
    fi
    echo ""
fi

# 輸入 OpenAI API Key
echo "請輸入您的 OpenAI API Key："
echo "（可從 https://platform.openai.com/api-keys 獲取）"
echo ""
read -p "OpenAI API Key: " OPENAI_KEY

# 驗證格式
if [ -z "$OPENAI_KEY" ]; then
    echo ""
    echo "❌ API Key 不能為空"
    exit 1
fi

if [[ ! $OPENAI_KEY == sk-* ]]; then
    echo ""
    echo "⚠️  警告：API Key 格式可能不正確（應以 sk- 開頭）"
    read -p "是否繼續？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 詢問是否設定 Gemini API Key（選用）
echo ""
echo "是否要設定 Google Gemini API Key？（選用）"
read -p "(y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "請輸入您的 Gemini API Key："
    read -p "Gemini API Key: " GEMINI_KEY
fi

# 創建 .env 檔案
cat > $ENV_FILE << EOF
# OpenAI API Key
# 從這裡獲取: https://platform.openai.com/api-keys
OPENAI_API_KEY=$OPENAI_KEY

EOF

# 如果有設定 Gemini Key，加入
if [ ! -z "$GEMINI_KEY" ]; then
    cat >> $ENV_FILE << EOF
# Google Gemini API Key (選用)
# 從這裡獲取: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=$GEMINI_KEY

EOF
fi

# 加入其他設定
cat >> $ENV_FILE << EOF
# 服務器設定
HOST=0.0.0.0
PORT=8000
FRONTEND_PORT=8080

# 資料庫設定
DB_PATH=./web/backend/chroma_db
COLLECTION_NAME=text_embedding_openai
EOF

# 設定檔案權限（只有擁有者可讀寫）
chmod 600 $ENV_FILE

echo ""
echo "===================================="
echo " ✅ .env 檔案創建成功！"
echo "===================================="
echo ""
echo "📁 檔案位置：$ENV_FILE"
echo "🔒 檔案權限：-rw------- (僅您可讀寫)"
echo ""
echo "📋 檔案內容："
echo "----------------------------------------"
cat $ENV_FILE | sed "s/\(OPENAI_API_KEY=sk-\).*/\1********************/"
echo "----------------------------------------"
echo ""
echo "💡 下一步："
echo "  ./start_all.sh    # 啟動系統"
echo ""

