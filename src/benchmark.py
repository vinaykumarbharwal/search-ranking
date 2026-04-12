"""
Performance benchmarking for search ranking system
"""
import time
import numpy as np
from typing import List, Dict
import pandas as pd
from predict import SearchRanker

class Benchmark:
    """Benchmark search performance"""
    
    def __init__(self, ranker: SearchRanker):
        self.ranker = ranker
        self.results = []
    
    def run_benchmark(self, queries: List[str], iterations: int = 10):
        """Run performance benchmark"""
        print(f"Running benchmark with {len(queries)} queries, {iterations} iterations...")
        
        for query in queries:
            latencies = []
            
            for _ in range(iterations):
                start = time.time()
                self.ranker.search(query)
                latencies.append((time.time() - start) * 1000)
            
            self.results.append({
                'query': query,
                'avg_latency_ms': np.mean(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'throughput_qps': 1000 / np.mean(latencies)
            })
        
        return pd.DataFrame(self.results)
    
    def print_report(self, df: pd.DataFrame):
        """Print benchmark report"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        print("\n" + "="*60)
        print(f"Overall Average Latency: {df['avg_latency_ms'].mean():.2f} ms")
        print(f"Overall P95 Latency: {df['p95_latency_ms'].mean():.2f} ms")
        print(f"Overall Throughput: {df['throughput_qps'].mean():.2f} queries/sec")
        print("="*60)

if __name__ == "__main__":
    ranker = SearchRanker()
    
    # Sample queries
    queries = [
        "python programming",
        "machine learning",
        "web development",
        "data science",
        "deep learning"
    ]
    
    benchmark = Benchmark(ranker)
    results = benchmark.run_benchmark(queries)
    benchmark.print_report(results)
