"""
Training script for LambdaMART ranking model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import ndcg_score
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from retrieval import HybridRetriever
from features import FeatureExtractor

class RankerTrainer:
    """Trainer for Learning-to-Rank model"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = None
        self.retriever = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare training data"""
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} query-document pairs")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Extract features for training"""
        print("Extracting features...")
        
        # Initialize feature extractor
        all_docs = df['document'].unique().tolist()
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.fit(all_docs)
        
        # Extract features for each query-document pair
        X_list = []
        y_list = []
        qid_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            features = self.feature_extractor.extract_query_doc_features(
                row['query'], row['document'], row.get('doc_id', idx)
            )
            
            X_list.append([features[name] for name in sorted(features.keys())])
            y_list.append(row['relevance'])
            qid_list.append(row['qid'])
        
        X = np.array(X_list)
        y = np.array(y_list)
        qid = np.array(qid_list)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, qid
    
    def train(self, X: np.ndarray, y: np.ndarray, qid: np.ndarray):
        """Train LambdaMART model"""
        print("Training LambdaMART model...")
        
        # Convert to DMatrix with query groups
        unique_qids = np.unique(qid)
        query_groups = [np.sum(qid == q) for q in unique_qids]
        
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(query_groups)
        
        # LambdaMART parameters
        params = {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@10',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.0,
            'tree_method': 'hist',
            'seed': 42,
            'verbosity': 1
        }
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        print("Training completed!")
        
        # Feature importance
        importance = self.model.get_score(importance_type='gain')
        print("\nTop 10 important features:")
        feature_names = self.feature_extractor.feature_names
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, score in sorted_importance:
            feat_idx = int(feat.replace('f', ''))
            if feat_idx < len(feature_names):
                print(f"  {feature_names[feat_idx]}: {score:.4f}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, qid: np.ndarray):
        """Evaluate model performance"""
        dtest = xgb.DMatrix(X)
        y_pred = self.model.predict(dtest)
        
        # Calculate NDCG per query
        unique_qids = np.unique(qid)
        ndcg_scores = []
        
        for q in unique_qids:
            mask = qid == q
            y_true_q = y[mask]
            y_pred_q = y_pred[mask]
            
            if len(y_true_q) > 1:
                # Sort by prediction
                sorted_idx = np.argsort(y_pred_q)[::-1]
                y_true_sorted = y_true_q[sorted_idx]
                
                # Calculate NDCG@10
                k = min(10, len(y_true_sorted))
                ndcg = ndcg_score([y_true_sorted[:k]], [y_true_sorted[:k]])
                ndcg_scores.append(ndcg)
        
        print(f"\nEvaluation Results:")
        print(f"  Average NDCG@10: {np.mean(ndcg_scores):.4f}")
        print(f"  Median NDCG@10: {np.median(ndcg_scores):.4f}")
        
        return np.mean(ndcg_scores)
    
    def save(self, model_path: str, features_path: str, scaler_path: str):
        """Save model and artifacts"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.model.save_model(model_path)
        self.feature_extractor.save(features_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {model_path}")
        print(f"Feature extractor saved to {features_path}")
        print(f"Scaler saved to {scaler_path}")

def generate_sample_data():
    """Generate sample training data"""
    queries = [
        "python programming", "machine learning", "data science",
        "web development", "deep learning", "artificial intelligence",
        "python tutorial", "java programming", "cloud computing"
    ]
    
    documents = [
        "Learn Python programming from basics to advanced with examples",
        "Complete machine learning course with Python and scikit-learn",
        "Data science bootcamp: pandas, numpy, matplotlib tutorial",
        "Web development with React and Node.js full stack guide",
        "Deep learning specialization: neural networks and TensorFlow",
        "Artificial intelligence fundamentals: search algorithms and logic",
        "Python for beginners: getting started with Python programming",
        "Java programming: object-oriented programming masterclass",
        "AWS cloud computing: architecture and deployment guide",
        "Machine learning algorithms explained with Python code",
        "Data visualization with Python and matplotlib",
        "React hooks tutorial: modern React development"
    ]
    
    data = []
    qid = 0
    
    for query in queries:
        qid += 1
        for doc_id, doc in enumerate(documents):
            # Calculate relevance based on keyword matching
            relevance = 0
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            
            overlap = len(query_words & doc_words)
            if overlap > 0:
                relevance = min(overlap, 3)
            
            # Add some noise and variations
            if query.lower() in doc.lower():
                relevance = max(relevance, 2)
            
            data.append({
                'qid': qid,
                'query': query,
                'document': doc,
                'doc_id': doc_id,
                'relevance': relevance
            })
    
    df = pd.DataFrame(data)
    df.to_csv('data/training_data.csv', index=False)
    print(f"Generated {len(df)} training samples")
    return df

if __name__ == "__main__":
    # Create data directory
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Generate or load data
    if not os.path.exists('data/training_data.csv'):
        df = generate_sample_data()
    else:
        df = pd.read_csv('data/training_data.csv')
    
    # Train model
    trainer = RankerTrainer()
    X, y, qid = trainer.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test, qid_train, qid_test = train_test_split(
        X, y, qid, test_size=0.2, random_state=42
    )
    
    # Train
    trainer.train(X_train, y_train, qid_train)
    
    # Evaluate
    trainer.evaluate(X_test, y_test, qid_test)
    
    # Save
    trainer.save('models/ranker.json', 'models/features.pkl', 'models/scaler.pkl')
