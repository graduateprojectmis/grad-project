"""
測試腳本
用於測試新版 API 的各項功能
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
import json
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

API_BASE_URL = "http://localhost:8000"


def test_health():
    """測試健康檢查"""
    console.print("\n[bold cyan]測試 1: 健康檢查[/bold cyan]")
    console.print("=" * 60)

    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        response.raise_for_status()
        data = response.json()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("項目", style="cyan")
        table.add_column("值", style="green")

        for key, value in data.items():
            table.add_row(key, str(value))

        console.print(table)
        console.print("[bold green]✅ 健康檢查通過[/bold green]")
        return True

    except Exception as e:
        console.print(f"[bold red]❌ 健康檢查失敗：{e}[/bold red]")
        return False


def test_search():
    """測試搜尋功能"""
    console.print("\n[bold cyan]測試 2: 語義搜尋[/bold cyan]")
    console.print("=" * 60)

    query = "如何配對 AirPods"

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/search", json={"query": query, "n_results": 2}
        )
        response.raise_for_status()
        data = response.json()

        console.print(f"[bold]查詢：[/bold]{query}")
        console.print(f"[bold]狀態：[/bold]{data['status']}")
        console.print(f"[bold]結果數量：[/bold]{len(data['results'])}\n")

        for i, result in enumerate(data["results"], 1):
            console.print(f"[bold cyan]結果 {i}:[/bold cyan]")
            console.print(result[:200] + "..." if len(result) > 200 else result)
            console.print()

        console.print("[bold green]✅ 搜尋測試通過[/bold green]")
        return True

    except Exception as e:
        console.print(f"[bold red]❌ 搜尋測試失敗：{e}[/bold red]")
        return False


def test_ask():
    """測試問答功能"""
    console.print("\n[bold cyan]測試 3: 智慧問答[/bold cyan]")
    console.print("=" * 60)

    question = "如何配對 AirPods 到 iPhone？"

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/ask", json={"question": question, "top_k": 1}
        )
        response.raise_for_status()
        data = response.json()

        console.print(f"[bold]問題：[/bold]{data['question']}")
        console.print(f"[bold]狀態：[/bold]{data['status']}\n")
        console.print(f"[bold green]答案：[/bold green]")
        console.print(data["answer"])
        console.print()

        if data.get("source_chunks"):
            console.print(f"[bold]來源片段數量：[/bold]{len(data['source_chunks'])}")

        console.print("[bold green]✅ 問答測試通過[/bold green]")
        return True

    except Exception as e:
        console.print(f"[bold red]❌ 問答測試失敗：{e}[/bold red]")
        if hasattr(e, "response") and e.response:
            try:
                error_detail = e.response.json()
                console.print(f"[yellow]錯誤詳情：{error_detail}[/yellow]")
            except:
                pass
        return False


def test_api_key_status():
    """測試 API Key 狀態"""
    console.print("\n[bold cyan]測試 4: API Key 狀態[/bold cyan]")
    console.print("=" * 60)

    try:
        response = requests.get(f"{API_BASE_URL}/api/admin/api-key/status")
        response.raise_for_status()
        data = response.json()

        console.print(f"[bold]API Key 存在：[/bold]{data['exists']}")
        console.print(f"[bold]遮罩值：[/bold]{data['masked']}")

        if data["exists"]:
            console.print("[bold green]✅ API Key 已設定[/bold green]")
        else:
            console.print("[bold yellow]⚠️  API Key 未設定[/bold yellow]")

        return True

    except Exception as e:
        console.print(f"[bold red]❌ API Key 狀態查詢失敗：{e}[/bold red]")
        return False


def main():
    """主函數"""
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]AirPods Q&A API 測試套件 v2.0[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    results = []

    # 執行所有測試
    results.append(("健康檢查", test_health()))
    results.append(("API Key 狀態", test_api_key_status()))
    results.append(("語義搜尋", test_search()))
    results.append(("智慧問答", test_ask()))

    # 顯示測試結果摘要
    console.print("\n[bold cyan]測試結果摘要[/bold cyan]")
    console.print("=" * 60)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("測試項目", style="cyan")
    table.add_column("結果", style="green")

    for name, result in results:
        status = "[green]✅ 通過[/green]" if result else "[red]❌ 失敗[/red]"
        table.add_row(name, status)

    console.print(table)

    # 統計
    passed = sum(1 for _, result in results if result)
    total = len(results)

    console.print(f"\n[bold]通過：{passed}/{total}[/bold]")

    if passed == total:
        console.print("[bold green]🎉 所有測試通過！[/bold green]")
    else:
        console.print("[bold yellow]⚠️  部分測試失敗，請檢查日誌[/bold yellow]")


if __name__ == "__main__":
    main()
