"""
Lightning AutoML - Automated Machine Learning for Lightning ML
Simple interface: ml_type, method, data, and automatic optimization
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
import warnings
import time
import pickle
import json
from dataclasses import dataclass, asdict
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, silhouette_score
import optuna
import gc

from lightning_ml import (
    LinearRegression,
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    SVMClassifier, SVMRegressor,
    KNNClassifier, KNNRegressor,
    KMeans, DBSCAN, AgglomerativeClustering,
    Apriori
)


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


class LightningAutoML:
    """
    AutoML for Lightning ML with simple interface
    
    Parameters:
    -----------
    ml_type : str
        "supervised" or "unsupervised"
    method : str
        "classification" (classification), "regression" (regression), "cluster", "apriori"
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like, optional
        Training labels (not needed for unsupervised)
    y_test : array-like, optional
        Test labels (not needed for unsupervised)
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
                 verbose: bool = True):
        
        # Validate inputs
        self.ml_type = ml_type.lower()
        self.method = method.lower()
        
        if self.ml_type not in ["supervised", "unsupervised"]:
            raise ValueError(f"ml_type must be 'supervised' or 'unsupervised', got '{ml_type}'")
        
        if self.method not in ["classification", "regression", "cluster", "apriori"]:
            raise ValueError(f"method must be 'classification', 'regression', 'cluster', or 'apriori', got '{method}'")
        
        # Validate ml_type and method combinations
        if self.ml_type == "supervised" and self.method not in ["classification", "regression"]:
            raise ValueError(f"For supervised learning, method must be 'classification' or 'regression'")
        
        if self.ml_type == "unsupervised" and self.method not in ["cluster", "apriori"]:
            raise ValueError(f"For unsupervised learning, method must be 'cluster' or 'apriori'")
        
        # Store data
        self.X_train = self._to_numpy(X_train)
        self.X_test = self._to_numpy(X_test)
        self.y_train = self._to_numpy(y_train) if y_train is not None else None
        self.y_test = self._to_numpy(y_test) if y_test is not None else None
        
        # Check y is provided for supervised
        if self.ml_type == "supervised" and (self.y_train is None or self.y_test is None):
            raise ValueError("y_train and y_test are required for supervised learning")
        
        # Configuration
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        
        # Device detection
        self.device = self._get_device()
        
        # Results
        self.results: List[ModelResult] = []
        self.best_model: Optional[ModelResult] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        
        # Preprocessing
        if self.method in ["classification", "regression"]:
            self._preprocess_data()
    
    def _to_numpy(self, data):
        """Convert to numpy array"""
        if data is None:
            return None
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return np.array(data)
    
    def _get_device(self) -> torch.device:
        """Auto-detect device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if self.verbose:
                print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device('cpu')
            if self.verbose:
                print(f"Using device: {device}")
        return device
    
    def _preprocess_data(self):
        """Preprocess training and test data"""
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Encode labels for classification
        if self.method == "classification":
            self.label_encoder = LabelEncoder()
            self.y_train = self.label_encoder.fit_transform(self.y_train)
            self.y_test = self.label_encoder.transform(self.y_test)
    
    def _get_models(self) -> Dict[str, Any]:
        """Get models based on method"""
        if self.method == "classification":
            return {
                "DecisionTree": DecisionTreeClassifier,
                "RandomForest": RandomForestClassifier,
                "Bagging": BaggingClassifier,
                "SVM": SVMClassifier,
                "KNN": KNNClassifier,
            }
        elif self.method == "regression":
            return {
                "LinearRegression": LinearRegression,
                "DecisionTree": DecisionTreeRegressor,
                "RandomForest": RandomForestRegressor,
                "Bagging": BaggingRegressor,
                "SVM": SVMRegressor,
                "KNN": KNNRegressor,
            }
        elif self.method == "cluster":
            return {
                "KMeans": KMeans,
                "DBSCAN": DBSCAN,
                "Agglomerative": AgglomerativeClustering,
            }
        elif self.method == "apriori":
            return {"Apriori": Apriori}
        return {}
    
    def _get_hyperparameter_space(self, model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space"""
        if self.method == "classification":
            if model_name == "DecisionTree":
                return {
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
            elif model_name == "RandomForest":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                }
            elif model_name == "Bagging":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 5, 50),
                    "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
                }
            elif model_name == "SVM":
                return {
                    "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
                    "epochs": trial.suggest_int("epochs", 100, 500),
                }
            elif model_name == "KNN":
                return {
                    "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
                }
        
        elif self.method == "regression":
            if model_name == "LinearRegression":
                return {}
            elif model_name == "DecisionTree":
                return {
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                }
            elif model_name == "RandomForest":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                }
            elif model_name == "Bagging":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 5, 50),
                }
            elif model_name == "SVM":
                return {
                    "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
                    "epochs": trial.suggest_int("epochs", 100, 500),
                }
            elif model_name == "KNN":
                return {
                    "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
                }
        
        elif self.method == "cluster":
            if model_name == "KMeans":
                return {
                    "n_clusters": trial.suggest_int("n_clusters", 2, 10),
                }
            elif model_name == "DBSCAN":
                return {
                    "eps": trial.suggest_float("eps", 0.1, 2.0),
                    "min_samples": trial.suggest_int("min_samples", 2, 10),
                }
            elif model_name == "Agglomerative":
                return {
                    "n_clusters": trial.suggest_int("n_clusters", 2, 10),
                }
        
        return {}
    
    def _evaluate_model(self, model_class, model_name: str, params: Dict[str, Any]) -> float:
        """Evaluate model with cross-validation using built-in score() method"""
        if self.method in ["classification", "regression"]:
            # Supervised learning
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state) \
                if self.method == "classification" else KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            scores = []
            for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
                y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
                
                # Create and train model
                model = model_class(**params, device=self.device)
                
                X_tr_t = torch.from_numpy(X_tr).float().to(self.device)
                y_tr_t = torch.from_numpy(y_tr).float().to(self.device)
                X_val_t = torch.from_numpy(X_val).float().to(self.device)
                y_val_t = torch.from_numpy(y_val).float().to(self.device)
                
                try:
                    model.fit(X_tr_t, y_tr_t, verbose=False)
                    # Use the model's built-in score() method
                    score = model.score(X_val_t, y_val_t)
                    scores.append(score)
                except Exception as e:
                    if self.verbose:
                        print(f"  Error in fold: {e}")
                    return -np.inf if self.method == "regression" else 0.0
            
            return np.mean(scores)
        
        elif self.method == "cluster":
            # Unsupervised clustering
            try:
                model = model_class(**params)
                X_t = torch.from_numpy(self.X_train).float().to(self.device)
                
                if model_name == "DBSCAN":
                    labels = model.fit_predict(X_t)
                else:
                    model.fit(X_t)
                    labels = model.predict(X_t)
                
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                
                # Silhouette score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(self.X_train, labels)
                else:
                    score = -1.0
                
                return score
            except Exception as e:
                if self.verbose:
                    print(f"  Error: {e}")
                return -1.0
        
        return 0.0
    
    def fit(self):
        """Train and optimize all models"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Lightning AutoML")
            print(f"ML Type: {self.ml_type.upper()} | Method: {self.method.upper()}")
            print(f"Device: {self.device}")
            print(f"{'='*60}\n")
        
        models = self._get_models()
        
        if self.method == "apriori":
            # Special case for Apriori
            if self.verbose:
                print("Training Apriori...")
            
            start_time = time.time()
            model = Apriori(min_support=0.1, min_confidence=0.5)
            
            # Convert to DataFrame if needed
            if isinstance(self.X_train, np.ndarray):
                X_df = pd.DataFrame(self.X_train)
            else:
                X_df = self.X_train
            
            model.fit(X_df)
            training_time = time.time() - start_time
            
            self.best_model = ModelResult(
                model_name="Apriori",
                model=model,
                score=1.0,
                cv_scores=[],
                cv_std=0.0,
                training_time=training_time,
                hyperparameters={"min_support": 0.1, "min_confidence": 0.5},
                device=str(self.device)
            )
            
            if self.verbose:
                print(f"Apriori training completed in {training_time:.2f}s")
            
            return self
        
        # Train each model with hyperparameter optimization
        for model_name, model_class in models.items():
            if self.verbose:
                print(f"\nOptimizing {model_name}...")
            
            start_time = time.time()
            
            def objective(trial):
                params = self._get_hyperparameter_space(model_name, trial)
                return self._evaluate_model(model_class, model_name, params)
            
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )
            
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False, timeout=self.time_budget / len(models))
            
            # Train final model with best params
            best_params = study.best_params
            
            try:
                model = model_class(**best_params, device=self.device)
                
                if self.method in ["classification", "regression"]:
                    X_t = torch.from_numpy(self.X_train).float().to(self.device)
                    y_t = torch.from_numpy(self.y_train).float().to(self.device)
                    model.fit(X_t, y_t, verbose=False)
                else:
                    X_t = torch.from_numpy(self.X_train).float().to(self.device)
                    model.fit(X_t)
                
                # Get CV scores
                cv_scores = [study.trials[i].value for i in range(min(self.cv_folds, len(study.trials)))]
                
                result = ModelResult(
                    model_name=model_name,
                    model=model,
                    score=study.best_value,
                    cv_scores=cv_scores,
                    cv_std=np.std(cv_scores) if cv_scores else 0.0,
                    training_time=time.time() - start_time,
                    hyperparameters=best_params,
                    device=str(self.device)
                )
                
                self.results.append(result)
                
                if self.verbose:
                    print(f"  Best score: {study.best_value:.4f}")
                    print(f"  Time: {result.training_time:.2f}s")
                    print(f"  Params: {best_params}")
            
            except Exception as e:
                if self.verbose:
                    print(f"  Failed to train {model_name}: {e}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Select best model
        if self.results:
            self.best_model = max(self.results, key=lambda x: x.score)
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Best Model: {self.best_model.model_name}")
                print(f"Score: {self.best_model.score:.4f} ± {self.best_model.cv_std:.4f}")
                print(f"Training Time: {self.best_model.training_time:.2f}s")
                print(f"{'='*60}\n")
        
        return self
    
    def predict(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:
        """Make predictions on test set or custom data"""
        if self.best_model is None:
            raise ValueError("No model trained! Call fit() first.")
        
        if X is None:
            X = self.X_test
        else:
            X = self._to_numpy(X)
            if self.scaler is not None:
                X = self.scaler.transform(X)
        
        X_t = torch.from_numpy(X).float().to(self.device)
        predictions = self.best_model.model.predict(X_t)
        
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Decode labels for classification
        if self.method == "classification" and self.label_encoder is not None:
            predictions = np.round(predictions).astype(int)
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate best model on test set using built-in score() method"""
        if self.best_model is None:
            raise ValueError("No model trained!")
        
        results = {}
        
        if self.method in ["classification", "regression"]:
            X_t = torch.from_numpy(self.X_test).float().to(self.device)
            y_t = torch.from_numpy(self.y_test).float().to(self.device)
            
            # Use model's built-in score method
            score = self.best_model.model.score(X_t, y_t)
            
            if self.method == "classification":
                results["accuracy"] = score
                if self.verbose:
                    print(f"Test Accuracy: {score:.4f}")
            else:
                results["r2"] = score
                # Also get predictions for additional metrics
                predictions = self.predict()
                results["mse"] = mean_squared_error(self.y_test, predictions)
                if self.verbose:
                    print(f"Test R²: {results['r2']:.4f}")
                    print(f"Test MSE: {results['mse']:.4f}")
        
        elif self.method == "cluster":
            predictions = self.predict()
            if len(np.unique(predictions)) > 1:
                results["silhouette"] = silhouette_score(self.X_test, predictions)
                if self.verbose:
                    print(f"Silhouette Score: {results['silhouette']:.4f}")
        
        return results
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get leaderboard of all models"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for r in sorted(self.results, key=lambda x: x.score, reverse=True):
            data.append({
                "Model": r.model_name,
                "Score": f"{r.score:.4f}",
                "CV_Std": f"{r.cv_std:.4f}",
                "Time(s)": f"{r.training_time:.2f}",
                "Device": r.device,
                "Hyperparameters": str(r.hyperparameters)
            })
        
        return pd.DataFrame(data)
    
    def save_model(self, filepath: str):
        """Save model and preprocessors"""
        if self.best_model is None:
            raise ValueError("No model to save!")
        
        save_data = {
            "ml_type": self.ml_type,
            "method": self.method,
            "model_name": self.best_model.model_name,
            "hyperparameters": self.best_model.hyperparameters,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "device": str(self.device),
        }
        
        # Save model weights
        self.best_model.model.save_model(f"{filepath}_model.pth")
        
        # Save metadata
        with open(f"{filepath}_meta.pkl", "wb") as f:
            pickle.dump(save_data, f)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.verbose:
                print("GPU cache cleared")
        gc.collect()
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get parameters of the best model"""
        if self.best_model is None:
            raise ValueError("No model trained!")
        
        return {
            "model_name": self.best_model.model_name,
            "hyperparameters": self.best_model.hyperparameters,
            "score": self.best_model.score,
            "cv_std": self.best_model.cv_std,
            "training_time": self.best_model.training_time,
            "device": self.best_model.device
        }


# Convenience functions
def auto_classification(X_train, X_test, y_train, y_test, **kwargs):
    """Quick classification training"""
    automl = LightningAutoML("supervised", "classification", X_train, X_test, y_train, y_test, **kwargs)
    automl.fit()
    return automl


def auto_regression(X_train, X_test, y_train, y_test, **kwargs):
    """Quick regression training"""
    automl = LightningAutoML("supervised", "regression", X_train, X_test, y_train, y_test, **kwargs)
    automl.fit()
    return automl


def auto_clustering(X_train, X_test, **kwargs):
    """Quick clustering"""
    automl = LightningAutoML("unsupervised", "cluster", X_train, X_test, **kwargs)
    automl.fit()
    return automl