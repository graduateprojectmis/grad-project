"""
批次向量化效能測試
比較逐一處理 vs 批次處理的 TTFT (Time To First Token / Time To First embedding)
"""

import time
import statistics
from typing import List, Dict, Any
from app.services.embedding_service import EmbeddingService


def generate_test_texts(count: int) -> List[str]:
    """生成測試用的文字資料"""
    test_texts = [
        "如何連接 AirPods 到 iPhone？",
        "AirPods Pro 的降噪功能怎麼開啟？",
        "如何檢查 AirPods 的電量？",
        "AirPods 無法配對怎麼辦？",
        "如何重置 AirPods？",
        "AirPods 的保固期是多久？",
        "AirPods 可以連接 Android 手機嗎？",
        "如何清潔 AirPods？",
        "AirPods Max 的特色功能有哪些？",
        "AirPods 的空間音訊功能如何使用？",
    ]
    # 循環使用測試文字直到達到指定數量
    return [test_texts[i % len(test_texts)] for i in range(count)]


def measure_sequential_embedding(
    service: EmbeddingService, texts: List[str]
) -> Dict[str, Any]:
    """
    測量逐一處理的效能
    
    Returns:
        包含 TTFT、總時間、各次請求時間的字典
    """
    times = []
    ttft = None  # Time To First Token (第一個嵌入向量的時間)
    
    total_start = time.perf_counter()
    
    for i, text in enumerate(texts):
        start = time.perf_counter()
        _ = service.generate_embedding(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            ttft = elapsed
    
    total_time = time.perf_counter() - total_start
    
    return {
        "method": "sequential",
        "ttft": ttft,
        "total_time": total_time,
        "avg_time_per_text": total_time / len(texts),
        "individual_times": times,
        "min_time": min(times),
        "max_time": max(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def measure_batch_embedding(
    service: EmbeddingService, texts: List[str], batch_size: int = None
) -> Dict[str, Any]:
    """
    測量批次處理的效能
    
    Returns:
        包含 TTFT、總時間的字典
    """
    batch_times = []
    ttft = None
    first_batch_done = False
    
    def progress_callback(done: int, total: int):
        nonlocal ttft, first_batch_done
        if not first_batch_done:
            ttft = time.perf_counter() - start_time
            first_batch_done = True
    
    start_time = time.perf_counter()
    
    # 使用批次處理
    _ = service.generate_embedding_batch(
        texts,
        batch_size=batch_size,
        delay_between_batches=0,  # 測試時不加延遲
        progress_callback=progress_callback,
    )
    
    total_time = time.perf_counter() - start_time
    
    # 如果只有一批，TTFT 就是總時間
    if ttft is None:
        ttft = total_time
    
    return {
        "method": "batch",
        "batch_size": batch_size or "default",
        "ttft": ttft,
        "total_time": total_time,
        "avg_time_per_text": total_time / len(texts),
    }


def measure_single_batch_embedding(
    service: EmbeddingService, texts: List[str]
) -> Dict[str, Any]:
    """
    測量一次性批次處理的效能（所有文字一次送出）
    
    Returns:
        包含 TTFT、總時間的字典
    """
    start_time = time.perf_counter()
    
    # 直接使用 generate_embedding 一次處理所有文字
    _ = service.generate_embedding(texts)
    
    total_time = time.perf_counter() - start_time
    
    return {
        "method": "single_batch",
        "ttft": total_time,  # 一次性批次的 TTFT 就是總時間
        "total_time": total_time,
        "avg_time_per_text": total_time / len(texts),
    }


def run_performance_test(
    text_counts: List[int] = None,
    provider: str = "openai",
    runs_per_test: int = 3,
):
    """
    執行效能測試
    
    Args:
        text_counts: 要測試的文字數量列表
        provider: 嵌入服務提供者
        runs_per_test: 每個測試執行的次數
    """
    if text_counts is None:
        text_counts = [5, 10, 20, 50]
    
    print("=" * 80)
    print(f"批次向量化效能測試 - Provider: {provider}")
    print("=" * 80)
    print(f"每個測試執行 {runs_per_test} 次取平均值")
    print()
    
    service = EmbeddingService(provider=provider)
    
    results = []
    
    for count in text_counts:
        print(f"\n{'='*60}")
        print(f"測試 {count} 個文字")
        print("=" * 60)
        
        texts = generate_test_texts(count)
        
        # 測試逐一處理
        sequential_results = []
        for run in range(runs_per_test):
            print(f"  逐一處理 - 執行 {run + 1}/{runs_per_test}...")
            result = measure_sequential_embedding(service, texts)
            sequential_results.append(result)
            time.sleep(0.5)  # 避免 API 限制
        
        # 計算平均值
        avg_sequential = {
            "method": "sequential",
            "text_count": count,
            "ttft": statistics.mean([r["ttft"] for r in sequential_results]),
            "total_time": statistics.mean([r["total_time"] for r in sequential_results]),
            "avg_time_per_text": statistics.mean([r["avg_time_per_text"] for r in sequential_results]),
        }
        
        # 測試一次性批次處理
        single_batch_results = []
        for run in range(runs_per_test):
            print(f"  一次性批次 - 執行 {run + 1}/{runs_per_test}...")
            result = measure_single_batch_embedding(service, texts)
            single_batch_results.append(result)
            time.sleep(0.5)
        
        avg_single_batch = {
            "method": "single_batch",
            "text_count": count,
            "ttft": statistics.mean([r["ttft"] for r in single_batch_results]),
            "total_time": statistics.mean([r["total_time"] for r in single_batch_results]),
            "avg_time_per_text": statistics.mean([r["avg_time_per_text"] for r in single_batch_results]),
        }
        
        # 測試分批處理（batch_size=10）
        batch_results = []
        batch_size = min(10, count)
        for run in range(runs_per_test):
            print(f"  分批處理 (batch_size={batch_size}) - 執行 {run + 1}/{runs_per_test}...")
            result = measure_batch_embedding(service, texts, batch_size=batch_size)
            batch_results.append(result)
            time.sleep(0.5)
        
        avg_batch = {
            "method": f"batch_{batch_size}",
            "text_count": count,
            "ttft": statistics.mean([r["ttft"] for r in batch_results]),
            "total_time": statistics.mean([r["total_time"] for r in batch_results]),
            "avg_time_per_text": statistics.mean([r["avg_time_per_text"] for r in batch_results]),
        }
        
        results.extend([avg_sequential, avg_single_batch, avg_batch])
        
        # 輸出結果
        print(f"\n  結果比較 ({count} 個文字):")
        print(f"  {'方法':<20} {'TTFT (秒)':<15} {'總時間 (秒)':<15} {'平均每個 (秒)':<15}")
        print(f"  {'-'*65}")
        print(f"  {'逐一處理':<20} {avg_sequential['ttft']:<15.4f} {avg_sequential['total_time']:<15.4f} {avg_sequential['avg_time_per_text']:<15.4f}")
        print(f"  {'一次性批次':<20} {avg_single_batch['ttft']:<15.4f} {avg_single_batch['total_time']:<15.4f} {avg_single_batch['avg_time_per_text']:<15.4f}")
        print(f"  {f'分批處理(size={batch_size})':<20} {avg_batch['ttft']:<15.4f} {avg_batch['total_time']:<15.4f} {avg_batch['avg_time_per_text']:<15.4f}")
        
        # 計算改善比例
        ttft_improvement = (avg_sequential['ttft'] - avg_single_batch['ttft']) / avg_sequential['ttft'] * 100
        total_improvement = (avg_sequential['total_time'] - avg_single_batch['total_time']) / avg_sequential['total_time'] * 100
        
        print(f"\n  一次性批次相對於逐一處理:")
        print(f"    TTFT 改善: {ttft_improvement:+.1f}% ({'更快' if ttft_improvement > 0 else '更慢'})")
        print(f"    總時間改善: {total_improvement:+.1f}% ({'更快' if total_improvement > 0 else '更慢'})")
    
    # 總結
    print("\n" + "=" * 80)
    print("總結")
    print("=" * 80)
    print("""
TTFT (Time To First Token) 分析:

1. 逐一處理：TTFT = 第一個請求的時間
   - 適合需要即時回應的場景
   - 總時間 = N 次 API 請求

2. 一次性批次處理：TTFT = 總時間（需等待所有結果）
   - 適合需要一次性處理大量資料的場景
   - 總時間明顯較短，但需要等待所有結果

3. 分批處理：TTFT = 第一批的時間
   - 平衡 TTFT 和總時間
   - 適合需要漸進式回應的場景
""")
    
    return results


if __name__ == "__main__":
    # 執行測試
    # 注意：大量文字會產生大量 API 費用和執行時間
    # 100 個文字 ≈ 約 100 次 API 請求（逐一處理）
    # 1000 個文字 ≈ 約 1000 次 API 請求（逐一處理）
    # 建議先用小規模測試，再逐步增加
    
    results = run_performance_test(
        text_counts=[100, 500, 1000, 5000, 10000],  # 可根據需求調整
        provider="openai",
        runs_per_test=3,  # 減少執行次數以節省時間和費用
    )
    
    # 如果要測試更大規模（謹慎使用，會產生大量費用）：
    # results = run_performance_test(
    #     text_counts=[10000, 50000, 100000, 500000, 1000000],
    #     provider="openai",
    #     runs_per_test=1,  # 大規模測試只執行一次
    # )
