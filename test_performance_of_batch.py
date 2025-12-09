"""
批次向量化效能測試
比較逐一處理 vs 批次處理的總時間
"""

import time
import statistics
from typing import List, Dict, Any
import matplotlib.pyplot as plt
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
        包含總時間、各次請求時間的字典
    """
    times = []
    
    total_start = time.perf_counter()
    
    for text in texts:
        start = time.perf_counter()
        _ = service.generate_embedding(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    total_time = time.perf_counter() - total_start
    
    return {
        "method": "sequential",
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
        包含總時間的字典
    """
    start_time = time.perf_counter()
    
    # 使用批次處理
    _ = service.generate_embedding_batch(
        texts,
        batch_size=batch_size,
        delay_between_batches=0,  # 測試時不加延遲
    )
    
    total_time = time.perf_counter() - start_time
    
    return {
        "method": "batch",
        "batch_size": batch_size or "default",
        "total_time": total_time,
        "avg_time_per_text": total_time / len(texts),
    }


def measure_single_batch_embedding(
    service: EmbeddingService, texts: List[str]
) -> Dict[str, Any]:
    """
    測量一次性批次處理的效能（所有文字一次送出）
    
    Returns:
        包含總時間的字典
    """
    start_time = time.perf_counter()
    
    # 直接使用 generate_embedding 一次處理所有文字
    _ = service.generate_embedding(texts)
    
    total_time = time.perf_counter() - start_time
    
    return {
        "method": "single_batch",
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
            "total_time": statistics.mean([r["total_time"] for r in batch_results]),
            "avg_time_per_text": statistics.mean([r["avg_time_per_text"] for r in batch_results]),
        }
        
        results.extend([avg_sequential, avg_single_batch, avg_batch])
        
        # 輸出結果
        print(f"\n  結果比較 ({count} 個文字):")
        print(f"  {'方法':<20} {'總時間 (秒)':<15} {'平均每個 (秒)':<15}")
        print(f"  {'-'*50}")
        print(f"  {'逐一處理':<20} {avg_sequential['total_time']:<15.4f} {avg_sequential['avg_time_per_text']:<15.4f}")
        print(f"  {'一次性批次':<20} {avg_single_batch['total_time']:<15.4f} {avg_single_batch['avg_time_per_text']:<15.4f}")
        print(f"  {f'分批處理(size={batch_size})':<20} {avg_batch['total_time']:<15.4f} {avg_batch['avg_time_per_text']:<15.4f}")
        
        # 計算改善比例
        total_improvement = (avg_sequential['total_time'] - avg_single_batch['total_time']) / avg_sequential['total_time'] * 100
        
        print(f"\n  一次性批次相對於逐一處理:")
        print(f"    總時間改善: {total_improvement:+.1f}% ({'更快' if total_improvement > 0 else '更慢'})")
    
    # 總結
    print("\n" + "=" * 80)
    print("總結")
    print("=" * 80)
    print("""
效能分析:

1. 逐一處理：總時間 = N 次 API 請求
   - 每個文字都需要一次 HTTP 請求
   - 網路延遲累積，效率較低

2. 一次性批次處理：總時間 = 1 次 API 請求
   - 所有文字一次送出
   - 大幅減少網路延遲，效率最高

3. 分批處理：總時間 = ceil(N / batch_size) 次 API 請求
   - 平衡記憶體使用和效能
   - 適合大量資料處理
""")
    
    return results


def generate_performance_chart(
    results: List[Dict[str, Any]],
    output_path: str = "batch_performance_chart.png",
    model_name: str = "OpenAI text-embedding-3-small",
):
    """
    根據測試結果生成效能圖表
    
    Args:
        results: run_performance_test 返回的結果列表
        output_path: 輸出圖片路徑
        model_name: 模型名稱（用於圖例）
    """
    # 分離不同方法的結果
    sequential_data = [r for r in results if r['method'] == 'sequential']
    single_batch_data = [r for r in results if r['method'] == 'single_batch']
    batch_data = [r for r in results if r['method'].startswith('batch_')]
    
    # 設定圖表
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 繪製三條線
    if sequential_data:
        text_counts = [r['text_count'] for r in sequential_data]
        times = [r['total_time'] for r in sequential_data]
        plt.plot(text_counts, times, marker='o', linewidth=2, markersize=8,
                 color='#d62728', label='逐一處理 (Sequential)')
    
    if single_batch_data:
        text_counts = [r['text_count'] for r in single_batch_data]
        times = [r['total_time'] for r in single_batch_data]
        plt.plot(text_counts, times, marker='s', linewidth=2, markersize=8,
                 color='#2ca02c', label='一次性批次 (Single Batch)')
    
    if batch_data:
        text_counts = [r['text_count'] for r in batch_data]
        times = [r['total_time'] for r in batch_data]
        batch_size = batch_data[0]['method'].split('_')[1]
        plt.plot(text_counts, times, marker='^', linewidth=2, markersize=8,
                 color='#1f77b4', label=f'分批處理 (Batch Size={batch_size})')
    
    # 設定標題和軸標籤
    plt.title(f'Embedding Time vs Text Count\nModel: {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Text Count (文字數量)', fontsize=12)
    plt.ylabel('Total Time (seconds)', fontsize=12)
    
    # 設定 X 軸刻度
    if sequential_data:
        plt.xticks([r['text_count'] for r in sequential_data])
    
    # 設定 Y 軸從 0 開始
    plt.ylim(bottom=0)
    
    # 添加圖例
    plt.legend(title='Method', loc='upper left')
    
    # 添加網格
    plt.grid(True, linestyle='-', alpha=0.7)
    
    # 調整布局
    plt.tight_layout()
    
    # 儲存圖片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n圖表已儲存至: {output_path}")
    
    plt.show()
    plt.close()


def run_batch_size_comparison(
    text_count: int = 1000,
    batch_sizes: List[int] = None,
    provider: str = "openai",
    runs_per_test: int = 2,
):
    """
    比較不同批次大小的效能
    
    Args:
        text_count: 測試的文字總數
        batch_sizes: 要測試的批次大小列表
        provider: 嵌入服務提供者
        runs_per_test: 每個測試執行的次數
    
    Returns:
        測試結果和圖表
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 25, 50, 100]
    
    print("=" * 80)
    print(f"批次大小效能比較測試 - Provider: {provider}")
    print(f"測試文字數量: {text_count}")
    print("=" * 80)
    print(f"每個測試執行 {runs_per_test} 次取平均值")
    print()
    
    service = EmbeddingService(provider=provider)
    texts = generate_test_texts(text_count)
    
    results = []
    
    for batch_size in batch_sizes:
        requests = (text_count + batch_size - 1) // batch_size  # ceil division
        print(f"\n測試 batch_size={batch_size} (預計 {requests} 次請求)...")
        
        batch_results = []
        for run in range(runs_per_test):
            print(f"  執行 {run + 1}/{runs_per_test}...")
            
            if batch_size == 1:
                # 逐一處理
                result = measure_sequential_embedding(service, texts)
            else:
                result = measure_batch_embedding(service, texts, batch_size=batch_size)
            
            batch_results.append(result)
            time.sleep(0.5)
        
        avg_time = statistics.mean([r['total_time'] for r in batch_results])
        results.append({
            'batch_size': batch_size,
            'requests': requests,
            'total_time': avg_time,
        })
        
        print(f"  平均時間: {avg_time:.3f} 秒")
    
    # 印出結果表格
    print("\n" + "=" * 60)
    print("測試結果")
    print("=" * 60)
    print(f"{'Batch Size':<15} {'Requests':<15} {'Total_Time(sec)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['batch_size']:<15} {r['requests']:<15,} {r['total_time']:<15.3f}")
    print("=" * 60)
    
    # 計算改善比例
    baseline = results[0]['total_time']  # batch_size=1
    best = min(results, key=lambda x: x['total_time'])
    improvement = (baseline - best['total_time']) / baseline * 100
    
    print(f"\n最佳批次大小: {best['batch_size']}")
    print(f"效能改善: {improvement:.1f}% (從 {baseline:.3f}s 降至 {best['total_time']:.3f}s)")
    
    return results


def generate_batch_size_chart(
    results: List[Dict[str, Any]],
    output_path: str = "batch_size_performance_chart.png",
    model_name: str = "OpenAI text-embedding-3-small",
):
    """
    根據批次大小測試結果生成效能圖表
    
    Args:
        results: run_batch_size_comparison 返回的結果列表
        output_path: 輸出圖片路徑
        model_name: 模型名稱（用於圖例）
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    batch_sizes = [r['batch_size'] for r in results]
    times = [r['total_time'] for r in results]
    
    plt.plot(batch_sizes, times, marker='o', linewidth=2, markersize=8,
             color='#1f77b4', label=model_name)
    
    plt.title('Total Embedding Time vs Batch Size', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Total Time (seconds)', fontsize=12)
    
    plt.xticks(batch_sizes)
    plt.ylim(bottom=0)
    
    plt.legend(title='Model', loc='upper right')
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n圖表已儲存至: {output_path}")
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    # ============================================================
    # 選擇要執行的測試類型
    # ============================================================
    
    # 測試類型 1: 比較不同批次大小的效能（生成類似你圖片的圖表）
    # 這會測試相同文字數量下，不同 batch_size 的效能差異
    
    print("執行批次大小比較測試...")
    batch_results = run_batch_size_comparison(
        text_count=100,  # 測試 100 個文字
        batch_sizes=[1, 8, 16, 32, 64, 128],  # 測試不同的批次大小
        provider="openai",
        runs_per_test=2,
    )
    
    # 生成 Batch Size vs Total Time 圖表
    generate_batch_size_chart(
        results=batch_results,
        output_path="/Users/luhsuehliang/Desktop/Grad-Project/batch_size_performance_chart.png",
        model_name="OpenAI text-embedding-3-small",
    )
    
    # ============================================================
    # 測試類型 2: 比較不同文字數量下的三種處理方法
    # 取消下方註解即可執行
    # ============================================================
    
    # print("\n執行文字數量比較測試...")
    # results = run_performance_test(
    #     text_counts=[10, 50, 100],
    #     provider="openai",
    #     runs_per_test=2,
    # )
    # 
    # # 生成 Text Count vs Total Time 圖表
    # generate_performance_chart(
    #     results=results,
    #     output_path="/Users/luhsuehliang/Desktop/Grad-Project/text_count_performance_chart.png",
    #     model_name="OpenAI text-embedding-3-small",
    # )
