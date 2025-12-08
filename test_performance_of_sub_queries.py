from app.services import QueryDecompositionService, DatabaseService, EmbeddingService
import matplotlib.pyplot as plt
import numpy as np

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang TC']
plt.rcParams['axes.unicode_minus'] = False

def compare_query_methods(query: str, n_results: int = 3, verbose: bool = True):
    """
    比較直接查詢與子查詢拆解的結果差異
    
    Args:
        query: 原始查詢
        n_results: 每個查詢返回的結果數量
        verbose: 是否輸出詳細資訊
    """
    # 初始化服務
    decomposer = QueryDecompositionService()
    db_service = DatabaseService()
    embedding_service = EmbeddingService()
    
    if verbose:
        print("=" * 80)
        print(f"原始查詢: {query}")
        print("=" * 80)
    
    # === 方法 1: 直接使用原始查詢 ===
    if verbose:
        print("\n【方法 1】直接使用原始查詢")
        print("-" * 40)
    
    query_embedding = embedding_service.generate_embedding(query)[0]
    direct_docs, direct_distances = db_service.query(query_embedding, n_results=n_results)
    
    if verbose:
        print(f"查詢結果 (前 {n_results} 個):")
        for i, (doc, dist) in enumerate(zip(direct_docs, direct_distances), 1):
            print(f"\n  [{i}] 距離: {dist:.4f}")
            print(f"      內容: {doc[:150]}..." if len(doc) > 150 else f"      內容: {doc}")
    
    # === 方法 2: 使用子查詢拆解 ===
    if verbose:
        print("\n" + "=" * 80)
        print("【方法 2】使用子查詢拆解 (Query Decomposition)")
        print("-" * 40)
    
    # 拆解問題
    sub_queries = decomposer.decompose_query(query)
    if verbose:
        print(f"拆解出 {len(sub_queries)} 個子查詢:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"  {i}. {sq}")
    
    # 對每個子查詢進行搜尋並合併結果
    all_results = {}  # 用 dict 去重，key 為文件內容，value 為最佳距離
    
    if verbose:
        print("\n子查詢結果:")
    for sq in sub_queries:
        if verbose:
            print(f"\n  子查詢: {sq}")
        sq_embedding = embedding_service.generate_embedding(sq)[0]
        sq_docs, sq_distances = db_service.query(sq_embedding, n_results=n_results)
        
        for doc, dist in zip(sq_docs, sq_distances):
            # 保留最佳（最小）距離
            if doc not in all_results or dist < all_results[doc]:
                all_results[doc] = dist
            if verbose:
                print(f"    - 距離: {dist:.4f} | {doc[:80]}...")
    
    # 按距離排序合併結果
    sorted_results = sorted(all_results.items(), key=lambda x: x[1])
    
    if verbose:
        print("\n" + "-" * 40)
        print(f"合併去重後的結果 (共 {len(sorted_results)} 個，顯示前 {n_results} 個):")
        for i, (doc, dist) in enumerate(sorted_results[:n_results], 1):
            print(f"\n  [{i}] 距離: {dist:.4f}")
            print(f"      內容: {doc[:150]}..." if len(doc) > 150 else f"      內容: {doc}")
    
    # === 比較分析 ===
    # 計算平均距離
    direct_avg_dist = sum(direct_distances) / len(direct_distances) if direct_distances else 0
    decomposed_top_dists = [dist for _, dist in sorted_results[:n_results]]
    decomposed_avg_dist = sum(decomposed_top_dists) / len(decomposed_top_dists) if decomposed_top_dists else 0
    
    # 計算最小距離
    direct_min_dist = min(direct_distances) if direct_distances else 0
    decomposed_min_dist = min(decomposed_top_dists) if decomposed_top_dists else 0
    
    if verbose:
        print("\n" + "=" * 80)
        print("【比較分析】")
        print("-" * 40)
        print(f"直接查詢 - 平均距離: {direct_avg_dist:.4f}")
        print(f"子查詢拆解 - 平均距離: {decomposed_avg_dist:.4f}")
        
        # 檢查結果重疊度
        direct_set = set(direct_docs)
        decomposed_set = set([doc for doc, _ in sorted_results[:n_results]])
        overlap = direct_set & decomposed_set
        
        print(f"\n結果重疊數量: {len(overlap)} / {n_results}")
        print(f"子查詢拆解獨有結果: {len(decomposed_set - direct_set)}")
        print(f"直接查詢獨有結果: {len(direct_set - decomposed_set)}")
        
        if decomposed_avg_dist < direct_avg_dist:
            improvement = ((direct_avg_dist - decomposed_avg_dist) / direct_avg_dist) * 100
            print(f"\n✅ 子查詢拆解方法距離更短（更準確），改善幅度: {improvement:.2f}%")
        elif decomposed_avg_dist > direct_avg_dist:
            degradation = ((decomposed_avg_dist - direct_avg_dist) / direct_avg_dist) * 100
            print(f"\n⚠️ 直接查詢方法距離更短，差異: {degradation:.2f}%")
        else:
            print("\n📊 兩種方法距離相同")
    
    return {
        "query": query,
        "direct": {
            "docs": direct_docs, 
            "distances": direct_distances, 
            "avg_distance": direct_avg_dist,
            "min_distance": direct_min_dist
        },
        "decomposed": {
            "docs": [doc for doc, _ in sorted_results], 
            "distances": [dist for _, dist in sorted_results], 
            "avg_distance": decomposed_avg_dist,
            "min_distance": decomposed_min_dist
        },
        "sub_queries": sub_queries,
        "improvement": ((direct_avg_dist - decomposed_avg_dist) / direct_avg_dist) * 100 if direct_avg_dist > 0 else 0
    }


def batch_compare_and_visualize(queries: list, n_results: int = 5, save_path: str = "./data/output/query_comparison.png"):
    """
    批次比較多組查詢並產生視覺化圖表
    
    Args:
        queries: 查詢列表
        n_results: 每個查詢返回的結果數量
        save_path: 圖表儲存路徑
    """
    print("=" * 80)
    print("🔍 開始批次查詢比較測試")
    print(f"   共 {len(queries)} 組查詢，每組返回 {n_results} 個結果")
    print("=" * 80)
    
    all_results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] 處理查詢: {query[:50]}...")
        result = compare_query_methods(query, n_results=n_results, verbose=False)
        all_results.append(result)
        print(f"   直接查詢平均距離: {result['direct']['avg_distance']:.4f}")
        print(f"   子查詢拆解平均距離: {result['decomposed']['avg_distance']:.4f}")
        print(f"   改善幅度: {result['improvement']:.2f}%")
    
    # === 視覺化 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 準備資料
    query_labels = [f"Q{i+1}" for i in range(len(queries))]
    direct_avg_dists = [r['direct']['avg_distance'] for r in all_results]
    decomposed_avg_dists = [r['decomposed']['avg_distance'] for r in all_results]
    direct_min_dists = [r['direct']['min_distance'] for r in all_results]
    decomposed_min_dists = [r['decomposed']['min_distance'] for r in all_results]
    improvements = [r['improvement'] for r in all_results]
    
    # 圖1: 平均距離比較 (分組長條圖)
    ax1 = axes[0, 0]
    x = np.arange(len(query_labels))
    width = 0.35
    bars1 = ax1.bar(x - width/2, direct_avg_dists, width, label='Direct Query', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, decomposed_avg_dists, width, label='Query Decomposition', color='#4ECDC4', alpha=0.8)
    
    # 新增平均線
    direct_mean = np.mean(direct_avg_dists)
    decomposed_mean = np.mean(decomposed_avg_dists)
    ax1.axhline(y=direct_mean, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Direct Query Avg ({direct_mean:.3f})')
    ax1.axhline(y=decomposed_mean, color='#4ECDC4', linestyle='--', linewidth=2, label=f'Query Decomposition Avg ({decomposed_mean:.3f})')
    
    ax1.set_xlabel('Query')
    ax1.set_ylabel('Average Distance')
    ax1.set_title('Average Distance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(query_labels)
    ax1.set_ylim(bottom=0.5)  # 設定y軸起始點為0.5
    ax1.legend(fontsize=7)
    ax1.grid(axis='y', alpha=0.3)
    
    # 在長條上標註數值
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # 圖2: 最小距離比較 (分組長條圖)
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width/2, direct_min_dists, width, label='Direct Query', color='#FF6B6B', alpha=0.8)
    bars4 = ax2.bar(x + width/2, decomposed_min_dists, width, label='Query Decomposition', color='#4ECDC4', alpha=0.8)
    
    # 新增平均線
    direct_min_mean = np.mean(direct_min_dists)
    decomposed_min_mean = np.mean(decomposed_min_dists)
    ax2.axhline(y=direct_min_mean, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Direct Query Avg ({direct_min_mean:.3f})')
    ax2.axhline(y=decomposed_min_mean, color='#4ECDC4', linestyle='--', linewidth=2, label=f'Query Decomposition Avg ({decomposed_min_mean:.3f})')
    
    ax2.set_xlabel('Query')
    ax2.set_ylabel('Minimum Distance')
    ax2.set_title('Minimum Distance Comparison (Best Match)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(query_labels)
    ax2.set_ylim(bottom=0.5)  # 設定y軸起始點為0.5
    ax2.legend(fontsize=7)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # 圖3: 改善幅度 (水平長條圖)
    ax3 = axes[1, 0]
    colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]
    bars5 = ax3.barh(query_labels, improvements, color=colors, alpha=0.8)
    ax3.set_xlabel('Improvement (%)')
    ax3.set_ylabel('Query')
    ax3.set_title('Improvement Rate by Query Decomposition')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(axis='x', alpha=0.3)
    
    for i, (bar, imp) in enumerate(zip(bars5, improvements)):
        ax3.annotate(f'{imp:.1f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5 if imp >= 0 else -35, 0), textcoords="offset points", 
                    ha='left' if imp >= 0 else 'right', va='center', fontsize=9)
    
    # 圖4: 綜合統計 (表格 + 餅圖)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 計算統計數據
    avg_improvement = np.mean(improvements)
    positive_count = sum(1 for imp in improvements if imp > 0)
    negative_count = len(improvements) - positive_count
    
    # 繪製餅圖
    pie_ax = fig.add_axes([0.58, 0.12, 0.18, 0.25])
    pie_colors = ['#2ECC71', '#E74C3C']
    pie_labels = ['Improved', 'Not Improved']
    pie_sizes = [positive_count, negative_count]
    if negative_count == 0:
        pie_sizes = [positive_count]
        pie_colors = ['#2ECC71']
        pie_labels = ['Improved']
    pie_ax.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    pie_ax.set_title('Improvement Distribution', fontsize=10)
    
    # 統計摘要文字
    summary_text = f"""
    ══════════════════════════════════
           Summary Statistics
    ══════════════════════════════════
    
    Total Queries: {len(queries)}
    
    Direct Query:
      • Avg Distance: {np.mean(direct_avg_dists):.4f}
      • Best Distance: {min(direct_min_dists):.4f}
    
    Query Decomposition:
      • Avg Distance: {np.mean(decomposed_avg_dists):.4f}
      • Best Distance: {min(decomposed_min_dists):.4f}
    
    Improvement:
      • Average: {avg_improvement:.2f}%
      • Improved: {positive_count}/{len(queries)}
      • Not Improved: {negative_count}/{len(queries)}
    ══════════════════════════════════
    """
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print(f"✅ 圖表已儲存至: {save_path}")
    print("=" * 80)
    
    # 印出查詢對照表
    print("\n📋 查詢對照表:")
    print("-" * 80)
    for i, query in enumerate(queries, 1):
        print(f"  Q{i}: {query}")
    
    # === 獨立輸出圖1、圖2、圖3 ===
    output_dir = "./data/output"
    
    # 圖1: 平均距離比較 (獨立輸出)
    fig1, ax1_single = plt.subplots(figsize=(10, 6))
    bars1_s = ax1_single.bar(x - width/2, direct_avg_dists, width, label='Direct Query', color='#FF6B6B', alpha=0.8)
    bars2_s = ax1_single.bar(x + width/2, decomposed_avg_dists, width, label='Query Decomposition', color='#4ECDC4', alpha=0.8)
    
    # 新增平均線
    direct_mean = np.mean(direct_avg_dists)
    decomposed_mean = np.mean(decomposed_avg_dists)
    ax1_single.axhline(y=direct_mean, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Direct Query Avg ({direct_mean:.3f})')
    ax1_single.axhline(y=decomposed_mean, color='#4ECDC4', linestyle='--', linewidth=2, label=f'Query Decomposition Avg ({decomposed_mean:.3f})')
    
    ax1_single.set_xlabel('Query', fontsize=12)
    ax1_single.set_ylabel('Average Distance', fontsize=12)
    ax1_single.set_title('Average Distance Comparison', fontsize=14)
    ax1_single.set_xticks(x)
    ax1_single.set_xticklabels(query_labels)
    ax1_single.set_ylim(bottom=0.5)  # 設定y軸起始點為0.5
    ax1_single.legend(fontsize=10)
    ax1_single.grid(axis='y', alpha=0.3)
    for bar in bars1_s:
        height = bar.get_height()
        ax1_single.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2_s:
        height = bar.get_height()
        ax1_single.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig1.savefig(f"{output_dir}/chart1_avg_distance.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"✅ 圖1 已儲存至: {output_dir}/chart1_avg_distance.png")
    
    # 圖2: 最小距離比較 (獨立輸出)
    fig2, ax2_single = plt.subplots(figsize=(10, 6))
    bars3_s = ax2_single.bar(x - width/2, direct_min_dists, width, label='Direct Query', color='#FF6B6B', alpha=0.8)
    bars4_s = ax2_single.bar(x + width/2, decomposed_min_dists, width, label='Query Decomposition', color='#4ECDC4', alpha=0.8)
    
    # 新增平均線
    direct_min_mean = np.mean(direct_min_dists)
    decomposed_min_mean = np.mean(decomposed_min_dists)
    ax2_single.axhline(y=direct_min_mean, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Direct Query Avg ({direct_min_mean:.3f})')
    ax2_single.axhline(y=decomposed_min_mean, color='#4ECDC4', linestyle='--', linewidth=2, label=f'Query Decomposition Avg ({decomposed_min_mean:.3f})')
    
    ax2_single.set_xlabel('Query', fontsize=12)
    ax2_single.set_ylabel('Minimum Distance', fontsize=12)
    ax2_single.set_title('Minimum Distance Comparison (Best Match)', fontsize=14)
    ax2_single.set_xticks(x)
    ax2_single.set_xticklabels(query_labels)
    ax2_single.set_ylim(bottom=0.5)  # 設定y軸起始點為0.5
    ax2_single.legend(fontsize=10)
    ax2_single.grid(axis='y', alpha=0.3)
    for bar in bars3_s:
        height = bar.get_height()
        ax2_single.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars4_s:
        height = bar.get_height()
        ax2_single.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig2.savefig(f"{output_dir}/chart2_min_distance.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ 圖2 已儲存至: {output_dir}/chart2_min_distance.png")
    
    # 圖3: 改善幅度 (獨立輸出)
    fig3, ax3_single = plt.subplots(figsize=(10, 6))
    colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]
    bars5_s = ax3_single.barh(query_labels, improvements, color=colors, alpha=0.8)
    ax3_single.set_xlabel('Improvement (%)', fontsize=12)
    ax3_single.set_ylabel('Query', fontsize=12)
    ax3_single.set_title('Improvement Rate by Query Decomposition', fontsize=14)
    ax3_single.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3_single.grid(axis='x', alpha=0.3)
    for i, (bar, imp) in enumerate(zip(bars5_s, improvements)):
        ax3_single.annotate(f'{imp:.1f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5 if imp >= 0 else -35, 0), textcoords="offset points", 
                    ha='left' if imp >= 0 else 'right', va='center', fontsize=10)
    plt.tight_layout()
    fig3.savefig(f"{output_dir}/chart3_improvement.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"✅ 圖3 已儲存至: {output_dir}/chart3_improvement.png")
    
    return all_results


if __name__ == "__main__":
    # 多組測試查詢
    test_queries = [
        "How to pair my airpods and how to open noise-canceling",
        "How to pair second-hand airpods to my device",
        "What is the battery life of AirPods Pro and how to check it",
        "How to reset AirPods and reconnect to iPhone",
        "AirPods microphone not working during calls",
        "How to switch AirPods between iPhone and Mac automatically",
        "AirPods spatial audio setup and compatible devices",
    ]
    
    # 執行批次比較並視覺化
    results = batch_compare_and_visualize(test_queries, n_results=5)
