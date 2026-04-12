"""
Comprehensive evaluation metrics for ranking system
"""
import numpy as np
from sklearn.metrics import average_precision_score
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RankingEvaluator:
    """Evaluate ranking performance"""
    
    def __init__(self):
        self.metrics_history = []
    
    def ndcg_at_k(self, y_true: List[float], y_pred: List[float], k: int = 10) -> float:
        """Calculate NDCG@K"""
        # Sort by prediction score
        sorted_idx = np.argsort(y_pred)[::-1]
        y_true_sorted = np.array(y_true)[sorted_idx][:k]
        
        # Calculate DCG
        dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(y_true_sorted))
        
        # Calculate IDCG
        ideal_sorted = sorted(y_true, reverse=True)[:k]
        idcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_sorted))
        
        return dcg / idcg if idcg > 0 else 0
    
    def map_at_k(self, y_true: List[float], y_pred: List[float], k: int = 10) -> float:
        """Calculate Mean Average Precision@K"""
        # Sort by prediction score
        sorted_idx = np.argsort(y_pred)[::-1][:k]
        y_true_sorted = np.array(y_true)[sorted_idx]
        
        # Calculate precision at each position
        precisions = []
        relevant_count = 0
        
        for i, rel in enumerate(y_true_sorted):
            if rel > 0:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        return np.mean(precisions) if precisions else 0
    
    def mrr(self, y_true: List[float], y_pred: List[float]) -> float:
        """Calculate Mean Reciprocal Rank"""
        sorted_idx = np.argsort(y_pred)[::-1]
        y_true_sorted = np.array(y_true)[sorted_idx]
        
        for i, rel in enumerate(y_true_sorted):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0
    
    def precision_at_k(self, y_true: List[float], y_pred: List[float], k: int = 10) -> float:
        """Calculate Precision@K"""
        sorted_idx = np.argsort(y_pred)[::-1][:k]
        y_true_sorted = np.array(y_true)[sorted_idx]
        return np.mean([1 if rel > 0 else 0 for rel in y_true_sorted])
    
    def recall_at_k(self, y_true: List[float], y_pred: List[float], k: int = 10) -> float:
        """Calculate Recall@K"""
        sorted_idx = np.argsort(y_pred)[::-1][:k]
        y_true_sorted = np.array(y_true)[sorted_idx]
        
        total_relevant = sum([1 if rel > 0 else 0 for rel in y_true])
        if total_relevant == 0:
            return 0
        
        retrieved_relevant = sum([1 if rel > 0 else 0 for rel in y_true_sorted])
        return retrieved_relevant / total_relevant
    
    def evaluate_all(self, y_true_list: List[List[float]], 
                    y_pred_list: List[List[float]], 
                    k: int = 10) -> Dict[str, float]:
        """Evaluate all metrics across multiple queries"""
        metrics = {
            f'NDCG@{k}': [],
            f'MAP@{k}': [],
            'MRR': [],
            f'Precision@{k}': [],
            f'Recall@{k}': []
        }
        
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            metrics[f'NDCG@{k}'].append(self.ndcg_at_k(y_true, y_pred, k))
            metrics[f'MAP@{k}'].append(self.map_at_k(y_true, y_pred, k))
            metrics['MRR'].append(self.mrr(y_true, y_pred))
            metrics[f'Precision@{k}'].append(self.precision_at_k(y_true, y_pred, k))
            metrics[f'Recall@{k}'].append(self.recall_at_k(y_true, y_pred, k))
        
        # Calculate averages
        results = {metric: np.mean(values) for metric, values in metrics.items()}
        results.update({f'{metric}_std': np.std(values) for metric, values in metrics.items()})
        
        self.metrics_history.append(results)
        return results
    
    def plot_metrics_comparison(self, baseline_scores: List[float], 
                                model_scores: List[float],
                                metric_name: str = "NDCG@10"):
        """Plot comparison between baseline and model"""
        plt.figure(figsize=(10, 6))
        
        data = pd.DataFrame({
            'Baseline': baseline_scores,
            'LambdaMART': model_scores
        })
        
        sns.boxplot(data=data)
        plt.title(f'{metric_name} Comparison: Baseline vs LambdaMART')
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        
        # Add statistical annotation
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(baseline_scores, model_scores)
        plt.text(0.5, 0.95, f'p-value: {p_value:.4f}', 
                transform=plt.gca().transAxes, ha='center')
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=100)
        plt.show()
    
    def print_detailed_report(self, results: Dict[str, float]):
        """Print detailed evaluation report"""
        print("\n" + "="*50)
        print("RANKING SYSTEM EVALUATION REPORT")
        print("="*50)
        
        for metric, value in results.items():
            if not metric.endswith('_std'):
                std_key = f"{metric}_std"
                std_value = results.get(std_key, 0)
                print(f"\n{metric}:")
                print(f"  Mean: {value:.4f}")
                print(f"  Std:  {std_value:.4f}")
        
        print("\n" + "="*50)
        
        # Interpretation
        if results.get('NDCG@10', 0) > 0.8:
            print("✅ Excellent ranking quality")
        elif results.get('NDCG@10', 0) > 0.6:
            print("✅ Good ranking quality")
        elif results.get('NDCG@10', 0) > 0.4:
            print("⚠️ Moderate ranking quality - needs improvement")
        else:
            print("❌ Poor ranking quality - significant improvement needed")

# Example usage
if __name__ == "__main__":
    # Simulated data
    y_true_list = [
        [3, 2, 1, 0, 0],  # Query 1
        [2, 1, 0, 0, 0],  # Query 2
        [3, 0, 1, 0, 2]   # Query 3
    ]
    
    y_pred_list = [
        [0.9, 0.8, 0.7, 0.1, 0.05],  # Model predictions
        [0.85, 0.75, 0.6, 0.2, 0.1],
        [0.95, 0.7, 0.65, 0.5, 0.3]
    ]
    
    evaluator = RankingEvaluator()
    results = evaluator.evaluate_all(y_true_list, y_pred_list, k=3)
    evaluator.print_detailed_report(results)
