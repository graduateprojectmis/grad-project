#!/bin/bash
# 快速測試腳本

set -e  # 遇到錯誤立即退出

echo "======================================"
echo "🧪 快速測試腳本"
echo "======================================"
echo ""

# 檢查參數
case "${1:-all}" in
    "all")
        echo "📋 執行所有測試..."
        python run_tests.py
        ;;
    "api")
        echo "🌐 測試 API 層..."
        pytest tests/test_api.py -v
        ;;
    "config")
        echo "⚙️  測試配置層..."
        pytest tests/test_config.py -v
        ;;
    "core")
        echo "🔧 測試核心功能..."
        pytest tests/test_core.py -v
        ;;
    "models")
        echo "📦 測試資料模型..."
        pytest tests/test_models.py -v
        ;;
    "services")
        echo "🔨 測試服務層..."
        pytest tests/test_services.py -v
        ;;
    "utils")
        echo "🛠  測試工具函數..."
        pytest tests/test_utils.py -v
        ;;
    "coverage")
        echo "📊 生成覆蓋率報告..."
        pytest --cov=app --cov-report=html --cov-report=term-missing
        echo ""
        echo "覆蓋率報告已生成: htmlcov/index.html"
        echo "開啟報告..."
        open htmlcov/index.html || xdg-open htmlcov/index.html || start htmlcov/index.html
        ;;
    "failed")
        echo "❌ 重新執行失敗的測試..."
        pytest --lf -v
        ;;
    "quick")
        echo "⚡ 快速測試（無覆蓋率）..."
        pytest -v
        ;;
    "help")
        echo "用法: $0 [選項]"
        echo ""
        echo "選項:"
        echo "  all       - 執行所有測試（預設）"
        echo "  api       - 只測試 API 層"
        echo "  config    - 只測試配置層"
        echo "  core      - 只測試核心功能"
        echo "  models    - 只測試資料模型"
        echo "  services  - 只測試服務層"
        echo "  utils     - 只測試工具函數"
        echo "  coverage  - 生成並開啟覆蓋率報告"
        echo "  failed    - 重新執行失敗的測試"
        echo "  quick     - 快速測試（無覆蓋率）"
        echo "  help      - 顯示此幫助訊息"
        ;;
    *)
        echo "❌ 未知選項: $1"
        echo "執行 '$0 help' 查看可用選項"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "✅ 測試完成"
echo "======================================"
