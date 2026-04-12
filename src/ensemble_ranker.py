import numpy as np

class EnsembleRanker:
    """Ensemble of multiple ranking models"""
    
    def __init__(self):
        self.models = {}
        
        try:
            import xgboost as xgb
            import lightgbm as lgb
            from catboost import CatBoostRanker
            
            # Model 1: XGBoost LambdaMART
            self.models['xgboost'] = xgb.XGBRanker(
                objective='rank:ndcg',
                learning_rate=0.05,
                max_depth=6,
                n_estimators=200
            )
            
            # Model 2: LightGBM Ranker
            self.models['lightgbm'] = lgb.LGBMRanker(
                objective='lambdarank',
                metric='ndcg',
                learning_rate=0.05,
                num_leaves=31,
                n_estimators=200
            )
            
            # Model 3: CatBoost Ranker
            self.models['catboost'] = CatBoostRanker(
                loss_function='YetiRank',
                custom_metric='NDCG',
                learning_rate=0.05,
                depth=6,
                iterations=200
            )
        except ImportError:
            print("Warning: Ensure lightgbm, catboost, and xgboost are installed.")
    
    def train(self, X_train, y_train, qid_train):
        """Train all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            if name == 'lightgbm':
                model.fit(X_train, y_train, group=qid_train)
            elif name == 'catboost':
                model.fit(X_train, y_train, group_id=qid_train)
            else:
                model.fit(X_train, y_train, qid=qid_train)
    
    def predict(self, X):
        """Ensemble prediction with weighted voting"""
        if not self.models: return np.zeros(len(X))
        
        predictions = []
        weights = {'xgboost': 0.4, 'lightgbm': 0.35, 'catboost': 0.25}
        
        for name, model in self.models.items():
            pred = model.predict(X)
            # Normalize predictions
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)
            predictions.append(pred * weights[name])
        
        return np.sum(predictions, axis=0)
