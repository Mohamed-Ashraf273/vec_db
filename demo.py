import time
import numpy as np
import tracemalloc
import os
from vec_db import VecDB, DIMENSION


tracemalloc.start()

print("\n" + "="*70)
print(" VecDB Performance Demo")
print("="*70 + "\n")

db = VecDB(db_size=1_000_000, index_file_path='index_1M', database_file_path='saved_db_1M.dat', new_db=False)
all_vectors = db.get_all_rows()

def get_accuracy(retrieved_indices, query_vector, top_k=5):
    """
    Calculate recall@k and quality ratio.
    Recall = exact matches with true top-k
    Quality = avg similarity of retrieved / avg similarity of true top-k
    """
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10)
    actual_similarities = np.dot(all_vectors, query_vector)
    top_k_indices = np.argpartition(actual_similarities, -top_k)[-top_k:]
    top_k_indices_set = set(top_k_indices)
    recall = sum(1 for idx in retrieved_indices if idx in top_k_indices_set) / top_k
    retrieved_sims = actual_similarities[retrieved_indices]
    true_top_sims = actual_similarities[top_k_indices]
    quality_ratio = np.mean(retrieved_sims) / np.mean(true_top_sims)
    
    return recall, quality_ratio

# Test multiple queries to get average
n_queries = 100
print(f"Running {n_queries} retrieval queries...\n")

query_times = []
memory_usages = []
recalls = []
quality_ratios = []

for i in range(n_queries):
    query = np.random.rand(64).astype('float32')
    query /= np.linalg.norm(query) + 1e-10
    
    tracemalloc.reset_peak()
    mem_before = tracemalloc.get_traced_memory()[0]
    
    start_time = time.time()
    retrieved_indices = db.retrieve(query=query, top_k=5)
    end_time = time.time()
    
    mem_after, peak_mem = tracemalloc.get_traced_memory()
    
    query_time = end_time - start_time
    memory_used = (mem_after - mem_before) / (1024 * 1024)  # MB
    
    query_times.append(query_time * 1000)  # Convert to ms
    memory_usages.append(memory_used)
    
    recall, quality = get_accuracy(retrieved_indices, query)
    recalls.append(recall)
    quality_ratios.append(quality)
    
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{n_queries}")

tracemalloc.stop()

# Calculate statistics
query_times = np.array(query_times)
memory_usages = np.array(memory_usages)
recalls = np.array(recalls)
quality_ratios = np.array(quality_ratios)

mean_time = np.mean(query_times)
p95_time = np.percentile(query_times, 95)
p99_time = np.percentile(query_times, 99)
qps = 1000 / mean_time

mean_memory = np.mean(memory_usages)
max_memory = np.max(memory_usages)

avg_recall = np.mean(recalls)
avg_quality = np.mean(quality_ratios)

# Get disk usage
db_size_mb = os.path.getsize('saved_db_1M.dat') / (1024 * 1024)
index_size_mb = 0
for root, dirs, files in os.walk('index_1M'):
    for file in files:
        index_size_mb += os.path.getsize(os.path.join(root, file)) / (1024 * 1024)

total_disk_mb = db_size_mb + index_size_mb
raw_size_mb = 1_000_000 * DIMENSION * 4 / (1024 * 1024)
compression_ratio = raw_size_mb / total_disk_mb

# Print Results
print("\n" + "="*70)
print(" PERFORMANCE RESULTS")
print("="*70 + "\n")

print(f"Database Configuration:")
print(f"  • Vectors:     1,000,000")
print(f"  • Dimensions:  {DIMENSION}")

print(f"Query Performance (top_k=5, {n_queries} queries):")
print(f"  • Mean latency:  {mean_time:6.2f} ms")
print(f"  • P95 latency:   {p95_time:6.2f} ms")
print(f"  • P99 latency:   {p99_time:6.2f} ms")
print(f"  • Throughput:    {qps:6.1f} queries/sec")
print(f"  • Recall@5:      {avg_recall:.4f} ({avg_recall*100:.2f}%)")
print(f"  • Quality:       {avg_quality:.4f} ({avg_quality*100:.2f}%)\n")

print(f"Memory Usage (per query):")
print(f"  • Mean memory:   {mean_memory:6.3f} MB")
print(f"  • Max memory:    {max_memory:6.3f} MB\n")

print(f"Disk Usage:")
print(f"  • Database:      {db_size_mb:8.2f} MB")
print(f"  • Index:         {index_size_mb:8.2f} MB")
print(f"  • Total:         {total_disk_mb:8.2f} MB")
print(f"  • Compression:   {compression_ratio:8.2f}x\n")

# Generate detailed report
report = f"""
{'='*70}
VecDB Performance Report
{'='*70}

DATABASE CONFIGURATION
  Vectors:           1,000,000
  Dimensions:        {DIMENSION}
  Database path:     saved_db_1M.dat
  Index path:        index_1M/

{'='*70}

QUERY PERFORMANCE (top_k=5)
  Queries tested:    {n_queries}
  
  Accuracy Metrics:
    Recall@5:        {avg_recall:.4f} ({avg_recall*100:.2f}%)
    Quality Ratio:   {avg_quality:.4f} ({avg_quality*100:.2f}%)
  
  Note: Recall = exact matches with true top-5
        Quality = retrieved similarity / optimal similarity
  
  Latency:
    Mean:            {mean_time:.2f} ms
    Median:          {np.median(query_times):.2f} ms
    Min:             {np.min(query_times):.2f} ms
    Max:             {np.max(query_times):.2f} ms
    Std Dev:         {np.std(query_times):.2f} ms
    
  Percentiles:
    P50:             {np.percentile(query_times, 50):.2f} ms
    P95:             {p95_time:.2f} ms
    P99:             {p99_time:.2f} ms
    
  Throughput:        {qps:.1f} queries/second

{'='*70}

MEMORY USAGE
  Per Query:
    Mean:            {mean_memory:.3f} MB
    Max:             {max_memory:.3f} MB
    Min:             {np.min(memory_usages):.3f} MB

{'='*70}

DISK STORAGE
  Database size:     {db_size_mb:.2f} MB
  Index size:        {index_size_mb:.2f} MB
  Total size:        {total_disk_mb:.2f} MB
  
  Raw data size:     {raw_size_mb:.2f} MB (uncompressed)
  Compression ratio: {compression_ratio:.2f}x
  Bytes per vector:  {(total_disk_mb * 1024 * 1024) / 1_000_000:.2f}

{'='*70}

PERFORMANCE SUMMARY

  🎯 Quality:  {'Excellent' if avg_quality > 0.98 else 'Very Good' if avg_quality > 0.95 else 'Good' if avg_quality > 0.90 else 'Fair'}
             (Retrieved {avg_quality*100:.2f}% as good as optimal)
             (Recall@5: {avg_recall:.4f})

  ⚡ Speed:   {'Excellent' if mean_time < 10 else 'Very Good' if mean_time < 20 else 'Good'}
             (Mean: {mean_time:.2f}ms, {qps:.1f} QPS)
  
  💾 Memory:  {'Excellent' if mean_memory < 5 else 'Very Good' if mean_memory < 10 else 'Good'}
             (Avg: {mean_memory:.2f}MB per query)
  
  🗜️  Storage: {'Excellent' if compression_ratio > 2.0 else 'Very Good' if compression_ratio > 1.5 else 'Good'}
             ({compression_ratio:.2f}x compression)

{'='*70}
"""

# Save report
with open('performance_report.txt', 'w') as f:
    f.write(report)

print("="*70)
print(f"\n✓ Detailed report saved to: performance_report.txt\n")
