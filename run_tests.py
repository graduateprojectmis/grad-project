"""
執行測試套件的腳本
"""

import sys
import subprocess
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """執行測試"""
    print("=" * 60)
    print("🧪 執行測試套件")
    print("=" * 60)

    # 檢查 pytest 是否安裝
    try:
        import pytest
    except ImportError:
        print("❌ pytest 未安裝！")
        print("正在安裝測試依賴...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"]
        )
        import pytest

    # 執行測試
    args = [
        "-v",  # 詳細輸出
        "--tb=short",  # 簡短的錯誤追蹤
        "--cov=app",  # 程式碼覆蓋率
        "--cov-report=term-missing",  # 顯示未覆蓋的行
        "--cov-report=html",  # 生成 HTML 報告
        "tests/",  # 測試目錄
    ]

    # 如果有命令列參數，使用它們
    if len(sys.argv) > 1:
        args = sys.argv[1:]

    print(f"\n執行命令: pytest {' '.join(args)}\n")

    # 執行測試
    exit_code = pytest.main(args)

    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ 所有測試通過！")
        print("📊 覆蓋率報告已生成：htmlcov/index.html")
    else:
        print("❌ 部分測試失敗")
    print("=" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
