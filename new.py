file -1
"""
LightningAutoML - Automated Machine Learning with Lightning ML
================================================================
Complete AutoML module with hyperparameter optimization, cross-validation,
and model comparison.
"""

import os
import time
import json
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
import inspect

from pathlib import Path

# Import Lightning ML models
from lightning_ml.regression import (
    LinearRegression, LogisticRegression, RidgeRegression, LassoRegression
)
from lightning_ml.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor
)
from lightning_ml.ensemble import (
    RandomForestClassifier, RandomForestRegressor, BaggingClassifier
)
from lightning_ml.svm import (
    SVMClassifier, SVMRegressor
)
from lightning_ml.neighbours import (
    KNNClassifier, KNNRegressor
)
from lightning_ml.cluster import (
    KMeans, DBSCAN
)

from lightning_ml.market_basket import Apriori

# Import Optuna suggest functions
from lightning_ml.optuna_optimizer import (
    suggest_linear_regression_params,
    suggest_logistic_regression_params,
    suggest_ridge_regression_params,
    suggest_lasso_regression_params,
    suggest_svm_classifier_params,
    suggest_svm_regressor_params,
    suggest_decision_tree_classifier_params,
    suggest_decision_tree_regressor_params,
    suggest_random_forest_classifier_params,
    suggest_random_forest_regressor_params,
    suggest_bagging_classifier_params,
    suggest_knn_classifier_params,
    suggest_knn_regressor_params,
    suggest_kmeans_params,
    suggest_dbscan_params,
    suggest_apriori_params
)

warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """Results from model training"""
    model_name: str
    model: Any
    score: float
    cv_scores: List[float]
    cv_std: float
    training_time: float
    hyperparameters: Dict[str, Any]
    device: str
    memory_used_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding model object)"""
        serializable_params = {}
        for k, v in self.hyperparameters.items():
            if isinstance(v, torch.device):
                serializable_params[k] = str(v)
            elif hasattr(v, '__class__') and 'torch' in str(type(v)):
                serializable_params[k] = str(v)
            else:
                serializable_params[k] = v
        
        return {
            'model_name': self.model_name,
            'score': self.score,
            'cv_scores': self.cv_scores,
            'cv_std': self.cv_std,
            'training_time': self.training_time,
            'hyperparameters': serializable_params,
            'device': self.device,
            'memory_used_mb': self.memory_used_mb
        }


class LightningAutoML:
    """
    Automated Machine Learning with Lightning ML
    
    Supports:
    - Supervised: Classification, Regression
    - Unsupervised: Clustering
    
    Features:
    - Automatic hyperparameter tuning with Optuna
    - GPU acceleration with automatic CPU fallback
    - Memory management and optimization
    - Cross-validation
    - Model leaderboard and comparison
    - Model persistence (save/load)
    """
    
    # Model registry
    MODELS = {
        'classification': {
            'LogisticRegression': (LogisticRegression, suggest_logistic_regression_params),
            'SVMClassifier': (SVMClassifier, suggest_svm_classifier_params),
            'DecisionTreeClassifier': (DecisionTreeClassifier, suggest_decision_tree_classifier_params),
            'RandomForestClassifier': (RandomForestClassifier, suggest_random_forest_classifier_params),
            'BaggingClassifier': (BaggingClassifier, suggest_bagging_classifier_params),
            'KNNClassifier': (KNNClassifier, suggest_knn_classifier_params),
        },
        'regression': {
            'LinearRegression': (LinearRegression, suggest_linear_regression_params),
            'RidgeRegression': (RidgeRegression, suggest_ridge_regression_params),
            'LassoRegression': (LassoRegression, suggest_lasso_regression_params),
            'SVMRegressor': (SVMRegressor, suggest_svm_regressor_params),
            'DecisionTreeRegressor': (DecisionTreeRegressor, suggest_decision_tree_regressor_params),
            'RandomForestRegressor': (RandomForestRegressor, suggest_random_forest_regressor_params),
            'KNNRegressor': (KNNRegressor, suggest_knn_regressor_params),
        },
        'clustering': {
            'KMeans': (KMeans, suggest_kmeans_params),
            'DBSCAN': (DBSCAN, suggest_dbscan_params),
        }
    }
    
    def __init__(self, 
                 ml_type: str,
                 method: str,
                 X_train: Union[np.ndarray, pd.DataFrame],
                 X_test: Union[np.ndarray, pd.DataFrame],
                 y_train: Optional[Union[np.ndarray, pd.Series]] = None,
                 y_test: Optional[Union[np.ndarray, pd.Series]] = None,
                 time_budget: float = 300.0,
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 verbose: bool = True,
                 force_cpu: bool = False,
                 auto_fallback: bool = True):
        """
        Initialize LightningAutoML
        
        Parameters:
        -----------
        ml_type : str
            Type of ML task: 'supervised' or 'unsupervised'
        method : str
            Method: 'classification', 'regression', or 'clustering'
        X_train : array-like
            Training features
        X_test : array-like
            Test features
        y_train : array-like, optional
            Training labels (required for supervised learning)
        y_test : array-like, optional
            Test labels (required for supervised learning)
        time_budget : float
            Maximum time in seconds for optimization
        n_trials : int
            Number of Optuna trials per model
        cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed
        verbose : bool
            Print progress
        force_cpu : bool
            Force CPU usage
        auto_fallback : bool
            Automatically fallback to CPU if GPU fails
        """
        # Validate inputs
        self._validate_inputs(ml_type, method, y_train, y_test)
        
        # Store configuration
        self.ml_type = ml_type.lower()
        self.method = method.lower()
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        self.force_cpu = force_cpu
        self.auto_fallback = auto_fallback
        
        # Convert data to numpy arrays
        self.X_train = self._to_numpy(X_train)
        self.X_test = self._to_numpy(X_test)
        self.y_train = self._to_numpy(y_train) if y_train is not None else None
        self.y_test = self._to_numpy(y_test) if y_test is not None else None
        
        # Device setup
        self.device = self._setup_device()
        
        # Results storage
        self.results: List[ModelResult] = []
        self.best_model: Optional[ModelResult] = None
        self.fitted = False
        
        # Get models for this method
        self.models_to_try = self.MODELS.get(self.method, {})
        
        if self.verbose:
            self._print_header()
    
    def _validate_inputs(self, ml_type, method, y_train, y_test):
        """Validate input parameters"""
        valid_ml_types = ['supervised', 'unsupervised']
        valid_methods = ['classification', 'regression', 'clustering']
        
        if ml_type.lower() not in valid_ml_types:
            raise ValueError(f"ml_type must be one of {valid_ml_types}")
        
        if method.lower() not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        
        if ml_type.lower() == 'supervised':
            if method.lower() == 'clustering':
                raise ValueError("Clustering is unsupervised, use ml_type='unsupervised'")
            if y_train is None or y_test is None:
                raise ValueError("y_train and y_test required for supervised learning")
        
        if ml_type.lower() == 'unsupervised':
            if method.lower() != 'clustering':
                raise ValueError("Only clustering is supported for unsupervised learning")
    
    def _to_numpy(self, data):
        """Convert data to numpy array"""
        if data is None:
            return None
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return np.array(data)
    
    def _setup_device(self):
        """Setup computation device"""
        if self.force_cpu:
            return 'cpu'
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _print_header(self):
        """Print AutoML header"""
        print("\n" + "="*70)
        print("⚡ LIGHTNING AutoML")
        print("="*70)
        print(f"ML Type: {self.ml_type.upper()}")
        print(f"Method: {self.method.upper()}")
        print(f"Device: {self.device.upper()}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features: {self.X_train.shape[1]}")
        print(f"Time budget: {self.time_budget}s")
        print(f"Trials per model: {self.n_trials}")
        print(f"CV Folds: {self.cv_folds}")
        print(f"Models to try: {len(self.models_to_try)}")
        print("="*70 + "\n")
    
    def _filter_model_params(self, params, model_class):
        """Filter parameters to only include those accepted by the model"""
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered = {k: v for k, v in params.items() if k in valid_params}
        return filtered
    
    def _get_cv_splitter(self):
        """Get appropriate cross-validation splitter"""
        if self.method == 'classification':
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
    
    def _calculate_cv_score(self, model, X, y, scoring):
        """Calculate cross-validation score"""
        cv_splitter = self._get_cv_splitter()
        scores = []
        
        for train_idx, val_idx in cv_splitter.split(X, y):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            if scoring == 'accuracy':
                score = accuracy_score(y_fold_val, y_pred)
            elif scoring == 'r2':
                score = r2_score(y_fold_val, y_pred)
            elif scoring == 'neg_mse':
                score = -mean_squared_error(y_fold_val, y_pred)
            else:
                score = 0.0
            
            scores.append(score)
        
        return scores
    
    def _optimize_model(self, model_name, model_class, suggest_func):
        """Optimize a single model"""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🚀 Optimizing: {model_name}")
            print(f"{'='*70}")
        
        start_time = time.time()
        
        # Determine scoring metric
        if self.method == 'classification':
            scoring = 'accuracy'
            direction = 'maximize'
        elif self.method == 'regression':
            scoring = 'r2'
            direction = 'maximize'
        else:  # clustering
            scoring = 'silhouette'
            direction = 'maximize'
        
        # Define objective function
        def objective(trial):
            try:
                params = suggest_func(trial)
                model_params = self._filter_model_params(params, model_class)
                
                if self.method == 'clustering':
                    # Clustering: fit on train, score on test
                    model = model_class(**model_params)
                    model.fit(self.X_train)
                    labels = model.predict(self.X_test)
                    
                    if len(set(labels)) < 2:
                        return -1.0
                    
                    score = silhouette_score(self.X_test, labels)
                else:
                    # Supervised: use cross-validation
                    model = model_class(**model_params)
                    cv_scores = self._calculate_cv_score(model, self.X_train, self.y_train, scoring)
                    score = np.mean(cv_scores)
                
                return score
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Trial failed: {e}")
                return -999999.0
        
        # Create and run study
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Calculate timeout per model
        timeout_per_model = self.time_budget / len(self.models_to_try)
        
        study.optimize(
            objective, 
            n_trials=self.n_trials, 
            timeout=timeout_per_model,
            show_progress_bar=self.verbose
        )
        
        # Get best parameters and train final model
        best_params = self._filter_model_params(study.best_params, model_class)
        best_model = model_class(**best_params)
        
        if self.method == 'clustering':
            best_model.fit(self.X_train)
            cv_scores = [study.best_value]
            cv_std = 0.0
        else:
            best_model.fit(self.X_train, self.y_train)
            cv_scores = self._calculate_cv_score(best_model, self.X_train, self.y_train, scoring)
            cv_std = np.std(cv_scores)
        
        # Calculate test score
        if self.method == 'clustering':
            test_labels = best_model.predict(self.X_test)
            if len(set(test_labels)) >= 2:
                test_score = silhouette_score(self.X_test, test_labels)
            else:
                test_score = -1.0
        elif self.method == 'classification':
            y_pred = best_model.predict(self.X_test)
            test_score = accuracy_score(self.y_test, y_pred)
        else:  # regression
            y_pred = best_model.predict(self.X_test)
            test_score = r2_score(self.y_test, y_pred)
        
        training_time = time.time() - start_time
        
        # Estimate memory usage
        memory_used = self._estimate_memory_usage(best_model)
        
        result = ModelResult(
            model_name=model_name,
            model=best_model,
            score=test_score,
            cv_scores=cv_scores,
            cv_std=cv_std,
            training_time=training_time,
            hyperparameters=best_params,
            device=self.device,
            memory_used_mb=memory_used
        )
        
        if self.verbose:
            print(f"\n✅ {model_name} Complete!")
            print(f"  • Best Score: {test_score:.4f}")
            print(f"  • CV Mean: {np.mean(cv_scores):.4f} (±{cv_std:.4f})")
            print(f"  • Training Time: {training_time:.2f}s")
            print(f"  • Memory: {memory_used:.2f} MB")
        
        return result
    
    def _estimate_memory_usage(self, model):
        """Estimate model memory usage in MB"""
        try:
            import sys
            size = sys.getsizeof(pickle.dumps(model))
            return size / (1024 * 1024)
        except:
            return 0.0
    
    def fit(self):
        """Fit all models and find the best one"""
        if self.verbose:
            print("🔍 Starting AutoML optimization...\n")
        
        total_start = time.time()
        
        for model_name, (model_class, suggest_func) in self.models_to_try.items():
            try:
                result = self._optimize_model(model_name, model_class, suggest_func)
                self.results.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"❌ {model_name} failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Sort results by score
        self.results.sort(key=lambda x: x.score, reverse=True)
        
        if self.results:
            self.best_model = self.results[0]
            self.fitted = True
        
        total_time = time.time() - total_start
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ AutoML Complete!")
            print(f"{'='*70}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Models Trained: {len(self.results)}")
            if self.best_model:
                print(f"Best Model: {self.best_model.model_name}")
                print(f"Best Score: {self.best_model.score:.4f}")
            print(f"{'='*70}\n")
        
        return self
    
    def predict(self, X):
        """Predict using the best model"""
        if not self.fitted or self.best_model is None:
            raise RuntimeError("Must call fit() before predict()")
        
        X = self._to_numpy(X)
        return self.best_model.model.predict(X)
    
    def get_leaderboard(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """Get model leaderboard"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results[:top_k]:
            row = {
                'Model': result.model_name,
                'Score': result.score,
                'CV Mean': np.mean(result.cv_scores),
                'CV Std': result.cv_std,
                'Time (s)': result.training_time,
                'Memory (MB)': result.memory_used_mb
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def print_summary(self):
        """Print detailed summary"""
        if not self.results:
            print("No results available. Call fit() first.")
            return
        
        print("\n" + "="*70)
        print("📊 AutoML SUMMARY")
        print("="*70)
        
        leaderboard = self.get_leaderboard()
        print("\n" + leaderboard.to_string(index=False))
        
        if self.best_model:
            print("\n" + "="*70)
            print("🏆 BEST MODEL")
            print("="*70)
            print(f"Model: {self.best_model.model_name}")
            print(f"Score: {self.best_model.score:.4f}")
            print(f"CV Mean: {np.mean(self.best_model.cv_scores):.4f}")
            print(f"CV Std: {self.best_model.cv_std:.4f}")
            print(f"\nHyperparameters:")
            for key, value in self.best_model.hyperparameters.items():
                print(f"  • {key}: {value}")
        
        print("\n" + "="*70 + "\n")
    
    def save_best_model(self, filepath: str):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model using pickle
        save_data = {
            'model': self.best_model.model,
            'model_name': self.best_model.model_name,
            'score': self.best_model.score,
            'hyperparameters': self.best_model.hyperparameters,
            'cv_scores': self.best_model.cv_scores,
            'cv_std': self.best_model.cv_std,
            'training_time': self.best_model.training_time,
            'method': self.method,
            'ml_type': self.ml_type,
            'device': str(self.device)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save metadata as JSON
        metadata = self.best_model.to_dict()
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"✅ Best model saved to {filepath}")
            print(f"   Metadata saved to {metadata_path}")

    # Fixed save_all_models method
    def save_all_models(self, directory: str):
        """Save all trained models"""
        if not self.results:
            raise ValueError("No models trained yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for result in self.results:
            # Save model
            model_path = directory / f"{result.model_name}.pkl"
            save_data = {
                'model': result.model,
                'model_name': result.model_name,
                'score': result.score,
                'hyperparameters': result.hyperparameters,
                'cv_scores': result.cv_scores,
                'cv_std': result.cv_std,
                'training_time': result.training_time,
                'method': self.method,
                'ml_type': self.ml_type,
                'device': result.device
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Save metadata
            metadata = result.to_dict()
            metadata_path = directory / f"{result.model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"✅ All {len(self.results)} models saved to {directory}")

    # Fixed load_model method
    def load_model(self, filepath: str):
        """Load a saved model"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.best_model = ModelResult(
            model_name=save_data['model_name'],
            model=save_data['model'],
            score=save_data['score'],
            cv_scores=save_data['cv_scores'],
            cv_std=save_data.get('cv_std', np.std(save_data['cv_scores'])),
            training_time=save_data.get('training_time', 0.0),
            hyperparameters=save_data['hyperparameters'],
            device=save_data.get('device', str(self.device))
        )
        
        self.method = save_data.get('method', self.method)
        self.ml_type = save_data.get('ml_type', self.ml_type)
        self.fitted = True
        
        if self.verbose:
            print(f"✅ Model loaded from: {filepath}")
            print(f"   Model: {self.best_model.model_name}")
            print(f"   Score: {self.best_model.score:.4f}")
        
        return self

    # Fixed export_results method
    def export_results(self, filepath: str):
        """Export all results to JSON"""
        if not self.results:
            raise ValueError("No results to export. Call fit() first.")
        
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine metric name based on method
        if self.method == 'classification':
            metric_name = 'accuracy'
        elif self.method == 'regression':
            metric_name = 'r2_score'
        else:
            metric_name = 'silhouette_score'
        
        export_data = {
            'config': {
                'ml_type': self.ml_type,
                'method': self.method,
                'time_budget': self.time_budget,
                'n_trials': self.n_trials,
                'cv_folds': self.cv_folds,
                'device': str(self.device),
                'random_state': self.random_state
            },
            'summary': {
                'total_models_trained': len(self.results),
                'best_model': self.best_model.model_name if self.best_model else None,
                'best_score': self.best_model.score if self.best_model else None,
                'metric_name': metric_name
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        if self.verbose:
            print(f"✅ Results exported to: {filepath}")
            

def calculate_optimal_time_budget(X_train, mode='medium'):
    """
    Calculate optimal time budget based on dataset size and mode
    
    Parameters:
    -----------
    X_train : array-like
        Training data
    mode : str
        'basic' (2-3 trials), 'medium' (5-6 trials), or 'full' (10 trials)
    
    Returns:
    --------
    time_budget : float
        Optimal time budget in seconds
    n_trials : int
        Number of trials per model
    """
    n_rows, n_cols = X_train.shape
    dataset_size = n_rows * n_cols
    
    # Define mode configurations
    mode_configs = {
        'basic': {'trials': 3, 'multiplier': 1.0},
        'medium': {'trials': 6, 'multiplier': 2.0},
        'full': {'trials': 10, 'multiplier': 4.0}
    }
    
    if mode not in mode_configs:
        mode = 'medium'
    
    config = mode_configs[mode]
    n_trials = config['trials']
    multiplier = config['multiplier']
    
    # Determine base time budget based on dataset size
    if dataset_size < 10000:  # Small
        base_time = 30
    elif dataset_size < 500000:  # Medium
        base_time = 60
    elif dataset_size < 20000000:  # Large
        base_time = 120
    else:  # Very Large
        base_time = 300
    
    time_budget = base_time * multiplier
    
    return time_budget, n_trials


# Example usage:
def create_automl_with_auto_budget(X_train, X_test, y_train, y_test, 
                                   ml_type, method, mode='medium'):
    """
    Create AutoML instance with automatically calculated time budget
    """
    time_budget, n_trials = calculate_optimal_time_budget(X_train, mode)
    
    print(f"📊 Dataset: {X_train.shape[0]} rows × {X_train.shape[1]} columns")
    print(f"⚙️  Mode: {mode.upper()}")
    print(f"⏱️  Time Budget: {time_budget}s")
    print(f"🔄 Trials: {n_trials} per model")
    
    automl = LightningAutoML(
        ml_type=ml_type,
        method=method,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        time_budget=time_budget,
        n_trials=n_trials,
        verbose=True
    )
    
    return automl


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

"""
LightningAutoML - Complete Example Usage
Testing Classification, Regression, and Clustering
"""

if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_diabetes, make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("\n" + "="*70)
    print("⚡ LightningAutoML - Example Usage")
    print("="*70)
    
    # =====================================================================
    # Example 1: Classification (Iris Dataset)
    # =====================================================================
    print("\n📊 Example 1: Classification (Iris Dataset)")
    print("-" * 70)
    
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize AutoML for classification
    automl_clf = LightningAutoML(
        ml_type='supervised',
        method='classification',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        time_budget=30,
        n_trials=10,
        cv_folds=3,
        verbose=True
    )
    
    # Fit all models
    automl_clf.fit()
    
    # Print summary
    automl_clf.print_summary()
    
    # Display leaderboard
    print("\n🏆 Classification Leaderboard:")
    print(automl_clf.get_leaderboard())
    
    # Save best model
    automl_clf.save_best_model("models/best_iris_model.pkl")
    
    # Make predictions
    iris_predictions = automl_clf.predict(X_test)
    print(f"\n✅ Sample predictions: {iris_predictions[:5]}")
    print(f"   Actual values: {y_test[:5]}")
    
    # Export results
    automl_clf.export_results("results/iris_automl_results.json")
    
    # =====================================================================
    # Example 2: Regression (Diabetes Dataset)
    # =====================================================================
    print("\n\n📊 Example 2: Regression (Diabetes Dataset)")
    print("-" * 70)
    
    # Load dataset
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize AutoML for regression
    automl_reg = LightningAutoML(
        ml_type='supervised',
        method='regression',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        time_budget=30,
        n_trials=10,
        cv_folds=3,
        verbose=True
    )
    
    # Fit all models
    automl_reg.fit()
    
    # Print summary
    automl_reg.print_summary()
    
    # Display leaderboard
    print("\n🏆 Regression Leaderboard:")
    print(automl_reg.get_leaderboard())
    
    # Save best model
    automl_reg.save_best_model("models/best_diabetes_model.pkl")
    
    # Make predictions
    diabetes_predictions = automl_reg.predict(X_test)
    print(f"\n✅ Sample predictions: {diabetes_predictions[:5]}")
    print(f"   Actual values: {y_test[:5]}")
    
    # Export results
    automl_reg.export_results("results/diabetes_automl_results.json")
    
    # =====================================================================
    # Example 3: Clustering (Synthetic Blobs)
    # =====================================================================
    print("\n\n📊 Example 3: Clustering (Synthetic Blobs)")
    print("-" * 70)
    
    # Generate synthetic data
    X, y = make_blobs(
        n_samples=300, 
        centers=3, 
        n_features=2, 
        random_state=42
    )
    X_train, X_test = train_test_split(
        X, test_size=0.2, random_state=42
    )
    
    # Initialize AutoML for clustering
    automl_cluster = LightningAutoML(
        ml_type='unsupervised',
        method='clustering',
        X_train=X_train,
        X_test=X_test,
        time_budget=30,
        n_trials=10,
        verbose=True
    )
    
    # Fit all models
    automl_cluster.fit()
    
    # Print summary
    automl_cluster.print_summary()
    
    # Display leaderboard
    print("\n🏆 Clustering Leaderboard:")
    print(automl_cluster.get_leaderboard())
    
    # Save best model
    automl_cluster.save_best_model("models/best_clustering_model.pkl")
    
    # Make predictions (cluster assignments)
    cluster_labels = automl_cluster.predict(X_test)
    print(f"\n✅ Sample cluster labels: {cluster_labels[:10]}")
    print(f"   Unique clusters: {len(set(cluster_labels))}")
    
    # Export results
    automl_cluster.export_results("results/clustering_automl_results.json")
    
    # =====================================================================
    # Summary of All Experiments
    # =====================================================================
    print("\n\n" + "="*70)
    print("📈 FINAL SUMMARY - All Experiments")
    print("="*70)
    
    print("\n1️⃣  Classification (Iris):")
    print(f"   Best Model: {automl_clf.best_model.model_name}")
    print(f"   Best Score: {automl_clf.best_model.score:.4f}")
    
    print("\n2️⃣  Regression (Diabetes):")
    print(f"   Best Model: {automl_reg.best_model.model_name}")
    print(f"   Best Score: {automl_reg.best_model.score:.4f}")
    
    print("\n3️⃣  Clustering (Blobs):")
    print(f"   Best Model: {automl_cluster.best_model.model_name}")
    print(f"   Best Score: {automl_cluster.best_model.score:.4f}")
    
    print("\n" + "="*70)
    print("✅ All experiments completed successfully!")
    print("="*70 + "\n")
    
    # =====================================================================
    # Optional: Load and use a saved model
    # =====================================================================
    print("\n" + "="*70)
    print("🔄 Bonus: Loading a Saved Model")
    print("="*70)

    # Load the SAME dataset that was used for training (Iris with 4 features)
    X_loaded, y_loaded = load_iris(return_X_y=True)
    X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = train_test_split(
        X_loaded, y_loaded, test_size=0.2, random_state=42
    )

    # Create a new AutoML instance with the correct dataset
    automl_loaded = LightningAutoML(
        ml_type='supervised',
        method='classification',
        X_train=X_train_loaded,
        X_test=X_test_loaded,
        y_train=y_train_loaded,
        y_test=y_test_loaded,
        verbose=True
    )

    # Load the previously saved model
    automl_loaded.load_model("models/best_iris_model.pkl")

    # Make predictions with loaded model
    loaded_predictions = automl_loaded.predict(X_test_loaded)
    print(f"\n✅ Predictions from loaded model: {loaded_predictions[:5]}")
    print(f"   Actual values: {y_test_loaded[:5]}")
    
    # Compare with the original iris predictions (use the correct variable from earlier)
    loaded_accuracy = accuracy_score(y_test_loaded, loaded_predictions)
    print(f"   Loaded model accuracy: {loaded_accuracy:.4f}")
    print(f"   Matches original best score: {loaded_accuracy == automl_clf.best_model.score}")
    print(f"   Match original predictions: {all(loaded_predictions == iris_predictions)}")

    print("\n" + "="*70 + "\n")