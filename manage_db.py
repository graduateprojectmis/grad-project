"""
資料庫管理腳本
用於管理 ChromaDB
"""
import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import argparse
from app.config import get_settings
from app.services import DatabaseService
from app.core.logger import setup_logger, get_logger
from rich.console import Console
from rich.table import Table

# 初始化
settings = get_settings()
setup_logger(level="INFO")
logger = get_logger(__name__)
console = Console()


def show_status():
    """顯示資料庫狀態"""
    console.print("\n[bold cyan]ChromaDB 狀態[/bold cyan]")
    console.print("=" * 60)
    
    try:
        db_service = DatabaseService()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("項目", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("資料庫路徑", settings.chroma_db_path)
        table.add_row("集合名稱", settings.chroma_collection_name)
        table.add_row("文件數量", str(db_service.count()))
        
        console.print(table)
        
        db_service.close()
        
    except Exception as e:
        console.print(f"[bold red]錯誤：{e}[/bold red]")


def clear_db():
    """清空資料庫"""
    console.print("\n[bold yellow]⚠️  警告：即將清空資料庫[/bold yellow]")
    
    response = console.input("確定要繼續嗎？(yes/no): ")
    
    if response.lower() != "yes":
        console.print("[yellow]已取消[/yellow]")
        return
    
    try:
        db_service = DatabaseService()
        db_service.clear()
        console.print("[bold green]✅ 資料庫已清空[/bold green]")
        db_service.close()
        
    except Exception as e:
        console.print(f"[bold red]錯誤：{e}[/bold red]")


def query_db(query_text: str, n_results: int = 5):
    """查詢資料庫"""
    console.print(f"\n[bold cyan]查詢：{query_text}[/bold cyan]")
    console.print("=" * 60)
    
    try:
        from app.services import EmbeddingService
        
        db_service = DatabaseService()
        embedding_service = EmbeddingService(provider="openai")
        
        # 生成查詢嵌入向量
        query_embedding = embedding_service.generate_embedding([query_text])[0]
        
        # 查詢
        documents, distances = db_service.query(query_embedding, n_results)
        
        console.print(f"[bold]找到 {len(documents)} 個結果[/bold]\n")
        
        for i, (doc, dist) in enumerate(zip(documents, distances), 1):
            console.print(f"[bold cyan]結果 {i} (距離: {dist:.4f}):[/bold cyan]")
            console.print(doc[:200] + "..." if len(doc) > 200 else doc)
            console.print()
        
        db_service.close()
        
    except Exception as e:
        console.print(f"[bold red]錯誤：{e}[/bold red]")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="ChromaDB 管理工具")
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # status 命令
    subparsers.add_parser("status", help="顯示資料庫狀態")
    
    # clear 命令
    subparsers.add_parser("clear", help="清空資料庫")
    
    # query 命令
    query_parser = subparsers.add_parser("query", help="查詢資料庫")
    query_parser.add_argument("text", help="查詢文字")
    query_parser.add_argument("-n", "--n-results", type=int, default=5, help="返回結果數量")
    
    args = parser.parse_args()
    
    if args.command == "status":
        show_status()
    elif args.command == "clear":
        clear_db()
    elif args.command == "query":
        query_db(args.text, args.n_results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
