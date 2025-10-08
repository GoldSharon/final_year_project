"""
Lightning AutoML - Automated Machine Learning with GPU Acceleration
====================================================================
FIXED VERSION - Addresses cross-validation and JSON serialization issues
"""

import torch
import numpy as np
import pandas as pd
import gc
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Union, Optional, List, Dict, Any, Tuple
from pathlib import Path
import json

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score, r2_score, mean_squared_error, 
    silhouette_score, davies_bouldin_score
)

import optuna
from optuna.samplers import TPESampler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# DATA CLASSES
# ============================================================================

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
        # Convert hyperparameters to JSON-serializable format
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


# ============================================================================
# GPU MEMORY MANAGER
# ============================================================================

class GPUMemoryManager:
    """Manages GPU memory and provides fallback strategies"""
    
    def __init__(self, device: torch.device, verbose: bool = True):
        self.device = device
        self.verbose = verbose
        self.is_cuda = device.type == 'cuda'
        self.oom_count = 0
        self.fallback_to_cpu = False
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics in MB"""
        if not self.is_cuda:
            return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        free = total - allocated
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total
        }
    
    def check_memory_available(self, required_mb: float = 100) -> bool:
        """Check if enough GPU memory is available"""
        if not self.is_cuda:
            return True
        
        stats = self.get_memory_stats()
        available = stats['free']
        
        if self.verbose:
            logger.debug(f"GPU Memory: {available:.1f}MB free / {stats['total']:.1f}MB total")
        
        return available > required_mb
    
    def clear_cache(self, aggressive: bool = False):
        """Clear GPU cache"""
        if self.is_cuda:
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
        gc.collect()
    
    def handle_oom(self) -> bool:
        """Handle OOM error, returns True if should retry on CPU"""
        self.oom_count += 1
        
        if self.verbose:
            logger.warning(f"⚠️  GPU Out of Memory (OOM #{self.oom_count})")
        
        # Try aggressive cleanup first
        self.clear_cache(aggressive=True)
        
        # After 2 OOM errors, suggest CPU fallback
        if self.oom_count >= 2:
            if self.verbose:
                logger.warning("⚠️  Multiple OOM errors detected. Switching to CPU.")
            self.fallback_to_cpu = True
            return True
        
        return False
    
    def get_optimal_batch_size(self, base_batch_size: int, data_size: int) -> int:
        """Adjust batch size based on available memory"""
        if not self.is_cuda or self.fallback_to_cpu:
            return base_batch_size
        
        stats = self.get_memory_stats()
        free_ratio = stats['free'] / stats['total']
        
        # Reduce batch size if memory is low
        if free_ratio < 0.2:
            adjusted = max(8, base_batch_size // 4)
        elif free_ratio < 0.4:
            adjusted = max(16, base_batch_size // 2)
        else:
            adjusted = base_batch_size
        
        # Ensure batch size doesn't exceed data size
        adjusted = min(adjusted, data_size)
        
        if adjusted != base_batch_size and self.verbose:
            logger.info(f"Adjusted batch size: {base_batch_size} → {adjusted} (free memory: {free_ratio*100:.1f}%)")
        
        return adjusted


# ============================================================================
# LIGHTNING AUTO ML
# ============================================================================

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
            'supervised' or 'unsupervised'
        method : str
            'classification', 'regression', or 'clustering'
        X_train, X_test : array-like
            Training and test features
        y_train, y_test : array-like, optional
            Training and test labels (not needed for clustering)
        time_budget : float
            Maximum time in seconds for optimization
        n_trials : int
            Number of Optuna trials
        cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed
        verbose : bool
            Print progress information
        force_cpu : bool
            Force CPU usage even if GPU is available
        auto_fallback : bool
            Automatically fallback to CPU on GPU OOM
        """
        
        # Validate inputs
        self._validate_inputs(ml_type, method)
        
        self.ml_type = ml_type.lower()
        self.method = method.lower()
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        self.auto_fallback = auto_fallback
        
        # Convert data to numpy arrays
        self.X_train = self._to_numpy(X_train)
        self.X_test = self._to_numpy(X_test)
        self.y_train = self._to_numpy(y_train) if y_train is not None else None
        self.y_test = self._to_numpy(y_test) if y_test is not None else None
        
        # Setup device
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize memory manager
        self.memory_manager = GPUMemoryManager(self.device, verbose)
        
        # Results storage
        self.results: List[ModelResult] = []
        self.best_model: Optional[Any] = None
        self.best_result: Optional[ModelResult] = None
        
        # Model registry
        self._setup_model_registry()
        
        if self.verbose:
            logger.info(f"🚀 Lightning AutoML Initialized")
            logger.info(f"   ML Type: {self.ml_type}")
            logger.info(f"   Method: {self.method}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Training samples: {len(self.X_train)}")
            logger.info(f"   Test samples: {len(self.X_test)}")
            if self.device.type == 'cuda':
                stats = self.memory_manager.get_memory_stats()
                logger.info(f"   GPU Memory: {stats['total']:.1f}MB total")
    
    def _validate_inputs(self, ml_type: str, method: str):
        """Validate input parameters"""
        valid_ml_types = ['supervised', 'unsupervised']
        valid_methods = ['classification', 'regression', 'clustering']
        
        if ml_type.lower() not in valid_ml_types:
            raise ValueError(f"ml_type must be one of {valid_ml_types}")
        
        if method.lower() not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        
        if ml_type.lower() == 'unsupervised' and method.lower() != 'clustering':
            raise ValueError("Unsupervised learning only supports 'clustering' method")
        
        if ml_type.lower() == 'supervised' and method.lower() == 'clustering':
            raise ValueError("Supervised learning does not support 'clustering' method")
    
    def _to_numpy(self, data) -> np.ndarray:
        """Convert data to numpy array"""
        if data is None:
            return None
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return np.asarray(data)
    
    def _setup_model_registry(self):
        """Setup model registry based on method"""
        # Import models dynamically to avoid circular imports
        from lightning_ml.regression import (
            LinearRegression, RidgeRegression, LassoRegression
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
        from lightning_ml.optuna_optimizer import (
            suggest_linear_regression_params,
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
            suggest_dbscan_params
        )
        
        if self.method == 'regression':
            self.model_registry = {
                'LinearRegression': (LinearRegression, suggest_linear_regression_params),
                'RidgeRegression': (RidgeRegression, suggest_ridge_regression_params),
                'LassoRegression': (LassoRegression, suggest_lasso_regression_params),
                'SVMRegressor': (SVMRegressor, suggest_svm_regressor_params),
                'DecisionTreeRegressor': (DecisionTreeRegressor, suggest_decision_tree_regressor_params),
                'RandomForestRegressor': (RandomForestRegressor, suggest_random_forest_regressor_params),
                'KNNRegressor': (KNNRegressor, suggest_knn_regressor_params)
            }
            self.metric_name = 'R²'
            self.metric_direction = 'maximize'
            
        elif self.method == 'classification':
            self.model_registry = {
                'DecisionTreeClassifier': (DecisionTreeClassifier, suggest_decision_tree_classifier_params),
                'RandomForestClassifier': (RandomForestClassifier, suggest_random_forest_classifier_params),
                'BaggingClassifier': (BaggingClassifier, suggest_bagging_classifier_params),
                'SVMClassifier': (SVMClassifier, suggest_svm_classifier_params),
                'KNNClassifier': (KNNClassifier, suggest_knn_classifier_params)
            }
            self.metric_name = 'Accuracy'
            self.metric_direction = 'maximize'
            
        elif self.method == 'clustering':
            self.model_registry = {
                'KMeans': (KMeans, suggest_kmeans_params),
                'DBSCAN': (DBSCAN, suggest_dbscan_params)
            }
            self.metric_name = 'Silhouette'
            self.metric_direction = 'maximize'
    
    def _filter_model_params(self, params: Dict, model_class) -> Dict:
        """Filter parameters to only include those accepted by the model"""
        import inspect
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self', 'args', 'kwargs'}
        
        # Exclude internal attributes and runtime state
        excluded = {
            'is_fitted', 'n_features_in_', 'classes_', '_sklearn_model',
            'n_samples_', 'feature_names_in_', 'estimators_', 'estimator_',
            'tree_', 'support_', 'dual_coef_', 'intercept_', 'coef_'
        }
        
        filtered = {}
        for k, v in params.items():
            if k in valid_params and k not in excluded:
                # Convert torch.device to string for storage
                if isinstance(v, torch.device):
                    filtered[k] = v
                else:
                    filtered[k] = v
        
        return filtered
    

    def _get_clean_params_for_cv(self, model) -> Dict:
        """Get clean parameters for cross-validation (without internal state)"""
        import inspect
        
        # Get only the constructor parameters
        sig = inspect.signature(type(model).__init__)
        valid_params = set(sig.parameters.keys()) - {'self', 'args', 'kwargs'}
        
        # Get current parameters
        params = model.get_params() if hasattr(model, 'get_params') else {}
        
        # List of internal attributes to exclude (runtime state)
        excluded_attrs = {
            'is_fitted', 'n_features_in_', 'classes_', '_sklearn_model',
            'n_samples_', 'feature_names_in_', 'estimators_', 'estimator_',
            'tree_', 'support_', 'dual_coef_', 'intercept_', 'coef_',
            'X_train_', 'y_train_', 'loss_history_', 'optimizer_',
            'model_', 'scaler_', '_encoder', 'labels_', 'cluster_centers_',
            'n_classes_', 'n_outputs_', 'max_features_', 'n_features_',
            'feature_importances_', 'oob_score_', 'oob_decision_function_'
        }
        
        # Only include parameters that are:
        # 1. Valid constructor parameters
        # 2. Not in the excluded list
        # 3. Not private attributes (starting with _)
        clean_params = {}
        for k, v in params.items():
            if k in valid_params and k not in excluded_attrs and not k.startswith('_'):
                clean_params[k] = v
        
        return clean_params
    
    def _evaluate_model(self, model, X_test, y_test) -> float:
        """Evaluate model on test set"""
        if self.method == 'clustering':
            labels = model.predict(X_test)
            if len(set(labels)) < 2:
                return -1.0
            return silhouette_score(X_test, labels)
        elif self.method == 'classification':
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)
        elif self.method == 'regression':
            y_pred = model.predict(X_test)
            return r2_score(y_test, y_pred)
    
    def _cross_validate(self, model, X, y) -> Tuple[float, List[float]]:
        """Perform cross-validation"""
        if self.method == 'clustering':
            # For clustering, use simple train/test splits
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []
            for train_idx, val_idx in kf.split(X):
                try:
                    X_val = X[val_idx]
                    # Get clean parameters for creating a new model instance
                    model_params = self._get_clean_params_for_cv(model)
                    model_copy = type(model)(**model_params)
                    model_copy.fit(X[train_idx])
                    labels = model_copy.predict(X_val)
                    if len(set(labels)) >= 2:
                        score = silhouette_score(X_val, labels)
                        scores.append(score)
                except Exception as e:
                    logger.warning(f"CV fold failed: {str(e)}")
                    continue
            return np.mean(scores) if scores else -1.0, scores
        else:
            if self.method == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'r2'
            
            try:
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=scoring)
                return scores.mean(), scores.tolist()
            except Exception as e:
                logger.warning(f"Cross-validation failed: {str(e)}")
                return 0.0, []
    
    def _optimize_model(self, model_name: str, model_class, suggest_func) -> Optional[ModelResult]:
        """Optimize a single model with Optuna"""
        if self.verbose:
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 Optimizing: {model_name}")
            logger.info(f"{'='*70}")
        
        start_time = time.time()
        current_device = self.device
        
        # Check if we should fallback to CPU
        if self.memory_manager.fallback_to_cpu and self.auto_fallback:
            current_device = torch.device('cpu')
            if self.verbose:
                logger.info(f"   Using CPU due to previous OOM errors")
        
        def objective(trial):
            try:
                # Clear memory before each trial
                self.memory_manager.clear_cache()
                
                # Get hyperparameters
                params = suggest_func(trial)
                params['device'] = current_device
                
                # Filter parameters
                model_params = self._filter_model_params(params, model_class)
                
                # Create and train model
                model = model_class(**model_params)
                
                if self.method == 'clustering':
                    model.fit(self.X_train)
                    labels = model.predict(self.X_train)
                    if len(set(labels)) < 2:
                        return -1.0
                    score = silhouette_score(self.X_train, labels)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_train)
                    if self.method == 'classification':
                        score = accuracy_score(self.y_train, y_pred)
                    else:
                        score = r2_score(self.y_train, y_pred)
                
                return score
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and self.auto_fallback:
                    should_fallback = self.memory_manager.handle_oom()
                    if should_fallback:
                        raise optuna.TrialPruned()
                raise e
        
        try:
            # Create study
            study = optuna.create_study(
                direction=self.metric_direction,
                sampler=TPESampler(seed=self.random_state)
            )
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.time_budget / len(self.model_registry),
                show_progress_bar=False
            )
            
            # Get best params
            best_params_raw = study.best_params.copy()
            best_params = self._filter_model_params(best_params_raw, model_class)
            best_params['device'] = current_device
            
            # Train best model
            best_model = model_class(**best_params)
            
            if self.method == 'clustering':
                best_model.fit(self.X_train)
            else:
                best_model.fit(self.X_train, self.y_train)
            
            # Evaluate
            test_score = self._evaluate_model(best_model, self.X_test, self.y_test)
            
            # Cross-validation
            if self.method == 'clustering':
                cv_mean, cv_scores = self._cross_validate(best_model, self.X_train, None)
            else:
                cv_mean, cv_scores = self._cross_validate(best_model, self.X_train, self.y_train)
            
            cv_std = np.std(cv_scores) if cv_scores else 0.0
            
            # Memory stats
            memory_used = 0.0
            if current_device.type == 'cuda':
                stats = self.memory_manager.get_memory_stats()
                memory_used = stats['allocated']
            
            training_time = time.time() - start_time
            
            # Convert device to string for storage
            params_for_storage = {}
            for k, v in best_params.items():
                if isinstance(v, torch.device):
                    params_for_storage[k] = str(v)
                else:
                    params_for_storage[k] = v
            
            result = ModelResult(
                model_name=model_name,
                model=best_model,
                score=test_score,
                cv_scores=cv_scores,
                cv_std=cv_std,
                training_time=training_time,
                hyperparameters=params_for_storage,
                device=str(current_device),
                memory_used_mb=memory_used
            )
            
            if self.verbose:
                logger.info(f"✅ {model_name} Complete!")
                logger.info(f"   Test {self.metric_name}: {test_score:.4f}")
                logger.info(f"   CV {self.metric_name}: {cv_mean:.4f} ± {cv_std:.4f}")
                logger.info(f"   Time: {training_time:.2f}s")
                logger.info(f"   Device: {current_device}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {str(e)}")
            return None
    
    def fit(self):
        """Main training method - trains all models in the registry"""
        if self.verbose:
            logger.info(f"\n{'#'*70}")
            logger.info(f"# Starting AutoML Training")
            logger.info(f"# Models to train: {len(self.model_registry)}")
            logger.info(f"# Time budget: {self.time_budget}s")
            logger.info(f"# Trials per model: {self.n_trials}")
            logger.info(f"{'#'*70}\n")
        
        start_time = time.time()
        self.results = []
        
        # Train each model
        for model_name, (model_class, suggest_func) in self.model_registry.items():
            result = self._optimize_model(model_name, model_class, suggest_func)
            
            if result is not None:
                self.results.append(result)
            
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > self.time_budget:
                if self.verbose:
                    logger.warning(f"⏰ Time budget exceeded ({elapsed:.1f}s / {self.time_budget}s)")
                break
        
        # Sort results by score
        self.results.sort(key=lambda x: x.score, reverse=True)
        
        # Set best model
        if self.results:
            self.best_result = self.results[0]
            self.best_model = self.best_result.model
            
            if self.verbose:
                logger.info(f"\n{'='*70}")
                logger.info(f"🏆 BEST MODEL: {self.best_result.model_name}")
                logger.info(f"   Test {self.metric_name}: {self.best_result.score:.4f}")
                logger.info(f"={'='*70}\n")
        else:
            logger.error("❌ No models were successfully trained!")
        
        return self
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get leaderboard of all models"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for r in sorted(self.results, key=lambda x: x.score, reverse=True):
            cv_mean = np.mean(r.cv_scores) if r.cv_scores else 0.0
            row = {
                "Rank": len(data) + 1,
                "Model": r.model_name,
                f"Test_{self.metric_name}": f"{r.score:.4f}",
                f"CV_{self.metric_name}": f"{cv_mean:.4f}",
                "CV_Std": f"{r.cv_std:.4f}",
                "Time(s)": f"{r.training_time:.2f}",
                "Device": r.device,
                "Memory(MB)": f"{r.memory_used_mb:.1f}" if r.memory_used_mb > 0 else "N/A"
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def predict(self, X):
        """Predict using best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call fit() first.")
        return self.best_model.predict(self._to_numpy(X))
    
    def save_best_model(self, filepath: str):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.best_model.save_model(str(filepath))
        
        # Save metadata with JSON-serializable format
        metadata = self.best_result.to_dict()
        
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            logger.info(f"✅ Best model saved to {filepath}")
            logger.info(f"   Metadata saved to {metadata_path}")
    
    def save_all_models(self, directory: str):
        """Save all trained models"""
        if not self.results:
            raise ValueError("No models trained yet. Call fit() first.")
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for result in self.results:
            model_path = directory / f"{result.model_name}.pth"
            result.model.save_model(str(model_path))
            
            # Save metadata
            metadata = result.to_dict()
            metadata_path = directory / f"{result.model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        if self.verbose:
            logger.info(f"✅ All {len(self.results)} models saved to {directory}")
    
    def get_model_by_name(self, model_name: str):
        """Get a specific model by name"""
        for result in self.results:
            if result.model_name == model_name:
                return result.model
        raise ValueError(f"Model '{model_name}' not found")
    
    def print_summary(self):
        """Print a comprehensive summary"""
        if not self.results:
            print("No results available. Run fit() first.")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 LIGHTNING AUTOML SUMMARY")
        print(f"{'='*70}")
        print(f"ML Type: {self.ml_type.upper()}")
        print(f"Method: {self.method.upper()}")
        print(f"Models Trained: {len(self.results)}")
        print(f"Metric: {self.metric_name}")
        print(f"\n{'='*70}")
        print(f"🏆 TOP 3 MODELS")
        print(f"{'='*70}")
        
        for i, result in enumerate(self.results[:3], 1):
            cv_mean = np.mean(result.cv_scores) if result.cv_scores else 0.0
            print(f"\n{i}. {result.model_name}")
            print(f"   Test {self.metric_name}: {result.score:.4f}")
            print(f"   CV {self.metric_name}: {cv_mean:.4f} ± {result.cv_std:.4f}")
            print(f"   Training Time: {result.training_time:.2f}s")
            print(f"   Device: {result.device}")
        
        print(f"\n{'='*70}\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example with synthetic data
    from sklearn.datasets import load_iris, load_diabetes, make_blobs
    from sklearn.model_selection import train_test_split
    
    print("Example 1: Classification")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    automl = LightningAutoML(
        ml_type='supervised',
        method='classification',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        time_budget=60,
        n_trials=10,
        verbose=True
    )
    
    automl.fit()
    automl.print_summary()
    print("\nLeaderboard:")
    print(automl.get_leaderboard())
    
    # Save best model
    automl.save_best_model("models/best_iris_model.pth")