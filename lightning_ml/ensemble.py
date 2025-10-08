
import torch
import numpy as np
from typing import Union, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from .base_model import BaseTreeModel, BaseSupervisedModel
from pathlib import Path
import pickle
import json
from typing import Union, Optional
from sklearn.ensemble import RandomForestClassifier as SklearnRFClassifier
from sklearn.ensemble import RandomForestRegressor as SklearnRFRegressor
import copy


from .tree import DecisionTreeRegressor,DecisionTreeClassifier


class RandomForestRegressor(BaseSupervisedModel):
    """
    Hybrid Random Forest Regressor
    Uses Sklearn's fast Cython implementation internally with PyTorch interface
    
    Args:
        n_estimators: Number of trees in forest
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        max_features: Features to consider ('sqrt', 'log2', int, None)
        criterion: Split criterion ('squared_error', 'absolute_error', 'poisson')
        bootstrap: Use bootstrap sampling
        max_samples: Maximum samples per tree
        n_jobs: Parallel jobs (-1 = all cores)
        random_state: Random seed
        device: Computation device (for tensor operations)
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str]] = 1.0,
                 criterion: str = 'squared_error',
                 bootstrap: bool = True,
                 max_samples: Optional[float] = None,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        
        super().__init__(device)
        
        # Store parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Initialize sklearn model
        self._sklearn_model = SklearnRFRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        self.feature_importances_ = None
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            sample_weight: Optional[np.ndarray] = None,
            verbose: bool = False) -> 'HybridRandomForestRegressor':
        """
        Fit random forest regressor using sklearn's fast implementation.
        
        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights
            verbose: Print progress
            
        Returns:
            self: Fitted model
        """
        # Convert to numpy for sklearn
        X_np = self._to_numpy(X) if isinstance(X, torch.Tensor) else X
        y_np = self._to_numpy(y) if isinstance(y, torch.Tensor) else y
        
        # Validate input
        X_tensor = self._to_tensor(X_np)
        y_tensor = self._to_tensor(y_np)
        self._validate_input(X_tensor, y_tensor)
        
        if verbose:
            print(f"Training Random Forest Regressor with {self.n_estimators} trees...")
            print(f"Dataset: {X_np.shape[0]} samples, {X_np.shape[1]} features")
        
        # Fit sklearn model
        self._sklearn_model.fit(X_np, y_np, sample_weight=sample_weight)
        
        # Extract sklearn attributes
        self.feature_importances_ = self._sklearn_model.feature_importances_
        
        # Update fit info
        self._update_fit_info(X_np)
        
        if verbose:
            print(f"✓ Training completed!")
        
        return self
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict values using sklearn's fast implementation.
        
        Args:
            X: Input features
            
        Returns:
            Predictions (numpy array)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to numpy
        X_np = self._to_numpy(X) if isinstance(X, torch.Tensor) else X
        
        # Use sklearn's fast prediction
        return self._sklearn_model.predict(X_np)
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor],
              sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Calculate R² score.
        
        Args:
            X: Test features
            y: True values
            sample_weight: Sample weights
            
        Returns:
            R² score
        """
        X_np = self._to_numpy(X) if isinstance(X, torch.Tensor) else X
        y_np = self._to_numpy(y) if isinstance(y, torch.Tensor) else y
        
        return self._sklearn_model.score(X_np, y_np, sample_weight=sample_weight)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params(deep=deep)
        params.update({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'criterion': self.criterion,
            'bootstrap': self.bootstrap,
            'max_samples': self.max_samples,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'n_features_in_': self.n_features_in_
        })
        return params
    
    def save_model(self, filepath: str):
        """Save model to file."""
        filepath = Path(filepath)
        
        # Save sklearn model
        sklearn_path = filepath.with_suffix('.sklearn.pkl')
        with open(sklearn_path, 'wb') as f:
            pickle.dump(self._sklearn_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata
        metadata = {
            'params': self.get_params(),
            'feature_importances': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'training_history': self._training_history,
            'model_type': 'HybridRandomForestRegressor'
        }
        
        metadata_path = filepath.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved: {sklearn_path}, {metadata_path}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        filepath = Path(filepath)
        
        # Load sklearn model
        sklearn_path = filepath.with_suffix('.sklearn.pkl')
        with open(sklearn_path, 'rb') as f:
            self._sklearn_model = pickle.load(f)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Restore attributes
        params = metadata['params']
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.feature_importances_ = np.array(metadata['feature_importances']) if metadata['feature_importances'] else None
        self._training_history = metadata.get('training_history', {'loss': [], 'epoch': []})
        
        print(f"✓ Model loaded from {filepath}")



class RandomForestClassifier(BaseSupervisedModel):
    """
    Hybrid Random Forest Classifier
    Uses sklearn's optimized Cython implementation internally with PyTorch-compatible interface.
    
    Args:
        n_estimators: Number of trees
        criterion: Splitting criterion ('gini', 'entropy', 'log_loss')
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples in leaf
        max_features: Number/features considered per split ('sqrt', 'log2', int, float)
        bootstrap: Whether bootstrap samples are used
        oob_score: Use out-of-bag samples to estimate generalization accuracy
        n_jobs: Parallel jobs (-1 = all cores)
        random_state: Random seed
        device: PyTorch device
    """

    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str, float]] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):

        super().__init__(device)

        # Store hyperparameters
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Initialize sklearn model
        self._sklearn_model = SklearnRFClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state
        )

        self.feature_importances_ = None
        self.classes_ = None
        self.n_classes_ = None

    # =====================================================================
    # Core Methods
    # =====================================================================

    def fit(self, X: Union[np.ndarray, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            sample_weight: Optional[np.ndarray] = None,
            verbose: bool = False) -> 'RandomForestClassifier':
        """
        Train random forest classifier using sklearn backend.
        """

        X_np = self._to_numpy(X) if isinstance(X, torch.Tensor) else X
        y_np = self._to_numpy(y) if isinstance(y, torch.Tensor) else y

        # Validation
        X_tensor = self._to_tensor(X_np)
        y_tensor = self._to_tensor(y_np)
        self._validate_input(X_tensor, y_tensor)

        if verbose:
            print(f"Training Random Forest Classifier with {self.n_estimators} trees...")
            print(f"Dataset: {X_np.shape[0]} samples, {X_np.shape[1]} features")

        # Fit sklearn model
        self._sklearn_model.fit(X_np, y_np, sample_weight=sample_weight)

        # Extract attributes
        self.feature_importances_ = self._sklearn_model.feature_importances_
        self.classes_ = self._sklearn_model.classes_
        self.n_classes_ = len(self.classes_)

        # Update model metadata
        self._update_fit_info(X_np)

        if verbose:
            print("✓ Training completed!")

        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        X_np = self._to_numpy(X) if isinstance(X, torch.Tensor) else X
        return self._sklearn_model.predict(X_np)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        X_np = self._to_numpy(X) if isinstance(X, torch.Tensor) else X
        return self._sklearn_model.predict_proba(X_np)

    def score(self, X: Union[np.ndarray, torch.Tensor],
              y: Union[np.ndarray, torch.Tensor],
              sample_weight: Optional[np.ndarray] = None) -> float:
        """Compute classification accuracy."""
        X_np = self._to_numpy(X) if isinstance(X, torch.Tensor) else X
        y_np = self._to_numpy(y) if isinstance(y, torch.Tensor) else y
        return self._sklearn_model.score(X_np, y_np, sample_weight=sample_weight)

    # =====================================================================
    # Parameter & Persistence Methods
    # =====================================================================

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return model hyperparameters and metadata."""
        params = super().get_params(deep=deep)
        params.update({
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'n_features_in_': self.n_features_in_,
            'n_classes_': self.n_classes_
        })
        return params

    def save_model(self, filepath: str):
        """Save both sklearn model and metadata."""
        filepath = Path(filepath)
        sklearn_path = filepath.with_suffix('.sklearn.pkl')
        meta_path = filepath.with_suffix('.meta.json')

        with open(sklearn_path, 'wb') as f:
            pickle.dump(self._sklearn_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            'params': self.get_params(),
            'feature_importances': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'training_history': self._training_history,
            'model_type': 'HybridRandomForestClassifier'
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Model saved: {sklearn_path}, {meta_path}")

    def load_model(self, filepath: str):
        """Load sklearn model and metadata."""
        filepath = Path(filepath)
        sklearn_path = filepath.with_suffix('.sklearn.pkl')
        meta_path = filepath.with_suffix('.meta.json')

        with open(sklearn_path, 'rb') as f:
            self._sklearn_model = pickle.load(f)

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        params = metadata['params']
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.feature_importances_ = np.array(metadata.get('feature_importances', []))
        self.classes_ = np.array(metadata.get('classes', []))
        self._training_history = metadata.get('training_history', {'loss': [], 'epoch': []})
        self.is_fitted = True

        print(f"✓ Model loaded from {filepath}")



"""
Optimized Bagging Ensemble Implementation
Fast GPU-accelerated Bootstrap Aggregating for Classification and Regression
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Any, List, Dict
import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BaggingBase:
    """
    Base class for Bagging with optimized parallel estimator training.
    All operations use PyTorch for performance.
    """
    
    def __init__(self,
                 base_estimator: Optional[Any] = None,
                 n_estimators: int = 10,
                 max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize Bagging ensemble.
        
        Args:
            base_estimator: Base estimator to use
            n_estimators: Number of estimators
            max_samples: Number/fraction of samples per estimator
            max_features: Number/fraction of features per estimator
            bootstrap: Whether to use bootstrap samples
            bootstrap_features: Whether to bootstrap features
            random_state: Random seed
            device: Computation device
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        self.device = device if device is not None else self._get_default_device()
        
        self.estimators_ = []
        self.estimators_features_ = []
        self.n_features_ = None
        self.is_fitted = False
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        
        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
                torch.cuda.manual_seed_all(random_state)
            logger.info(f"Random state set to {random_state}")
    
    def _get_default_device(self) -> torch.device:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("CUDA available - using GPU acceleration")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("MPS available - using Apple Silicon GPU")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for computation")
        return device
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor], 
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert input data to PyTorch tensor."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=dtype)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device, dtype=dtype)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)
    
    def _validate_input(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Validate input data shapes and types."""
        if X.dim() == 1:
            raise ValueError("X must be 2D array (n_samples, n_features)")
        
        if y is not None:
            if len(X) != len(y):
                raise ValueError(f"X and y must have same number of samples. "
                               f"Got X: {len(X)}, y: {len(y)}")
    
    def _make_estimator(self):
        """Create a copy of base estimator."""
        if self.base_estimator is None:
            raise NotImplementedError("Must provide base_estimator or override _make_estimator")
        return copy.deepcopy(self.base_estimator)
    
    def _get_n_samples(self, n_samples_total: int) -> int:
        """Calculate number of samples for each estimator."""
        if isinstance(self.max_samples, int):
            return min(self.max_samples, n_samples_total)
        else:  # float
            return max(1, int(self.max_samples * n_samples_total))
    
    def _get_n_features(self, n_features_total: int) -> int:
        """Calculate number of features for each estimator."""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features_total)
        else:  # float
            return max(1, int(self.max_features * n_features_total))
    
    def _generate_sample_indices(self, n_samples: int, n_samples_draw: int, 
                                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate sample indices (with or without replacement)."""
        if self.bootstrap:
            indices = torch.randint(0, n_samples, (n_samples_draw,), 
                                   device=self.device, generator=generator)
        else:
            indices = torch.randperm(n_samples, device=self.device, 
                                    generator=generator)[:n_samples_draw]
        return indices
    
    def _generate_feature_indices(self, n_features: int, n_features_draw: int,
                                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate feature indices (with or without replacement)."""
        if self.bootstrap_features:
            indices = torch.randint(0, n_features, (n_features_draw,), 
                                   device=self.device, generator=generator)
        else:
            indices = torch.randperm(n_features, device=self.device, 
                                    generator=generator)[:n_features_draw]
        return torch.sort(indices)[0]
    
    def _fit_estimator(self, estimator_idx: int, X: torch.Tensor, y: torch.Tensor,
                      generator: Optional[torch.Generator] = None):
        """Fit a single estimator."""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        estimator = self._make_estimator()
        
        # Generate indices
        n_samples_draw = self._get_n_samples(n_samples)
        sample_indices = self._generate_sample_indices(n_samples, n_samples_draw, generator)
        
        n_features_draw = self._get_n_features(n_features)
        feature_indices = self._generate_feature_indices(n_features, n_features_draw, generator)
        
        # Extract subset
        X_subset = X[sample_indices][:, feature_indices]
        y_subset = y[sample_indices]
        
        # Fit estimator
        estimator.fit(X_subset, y_subset)
        
        return estimator, feature_indices
    
    def _update_fit_info(self, X: torch.Tensor):
        """Update metadata after fitting."""
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.is_fitted = True
        logger.info(f"Model fitted: n_samples={self.n_samples_seen_}, n_features={self.n_features_in_}")
    
    def fit(self, X: Union[torch.Tensor, np.ndarray], 
            y: Union[torch.Tensor, np.ndarray],
            verbose: bool = False):
        """
        Fit bagging ensemble.
        
        Args:
            X: Training features
            y: Training targets
            verbose: Print progress
            
        Returns:
            self: Fitted model
        """
        # Convert to tensors
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y, dtype=torch.float32)
        
        self._validate_input(X, y)
        
        self.n_features_ = X.shape[1]
        
        # Create generators for reproducibility
        generators = []
        if self.random_state is not None:
            for i in range(self.n_estimators):
                gen = torch.Generator(device=self.device)
                gen.manual_seed(self.random_state + i)
                generators.append(gen)
        else:
            generators = [None] * self.n_estimators
        
        # Train estimators
        results = []
        for i in range(self.n_estimators):
            result = self._fit_estimator(i, X, y, generators[i])
            results.append(result)
            
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Trained {i+1}/{self.n_estimators} estimators")
                print(f"Trained {i+1}/{self.n_estimators} estimators")
        
        # Unpack results
        self.estimators_ = [r[0] for r in results]
        self.estimators_features_ = [r[1] for r in results]
        
        self._update_fit_info(X)
        
        if verbose:
            logger.info(f"Bagging ensemble fitted with {self.n_estimators} estimators")
            print(f"Bagging ensemble fitted with {self.n_estimators} estimators")
        
        return self
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            'device': str(self.device),
            'is_fitted': self.is_fitted,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'random_state': self.random_state,
            'n_features_': self.n_features_,
            'n_features_in_': self.n_features_in_,
            'n_samples_seen_': self.n_samples_seen_
        }
        return params.copy() if deep else params
    
    def set_params(self, **params):
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Move feature indices to new device
        self.estimators_features_ = [
            feat.to(self.device) if isinstance(feat, torch.Tensor) else feat
            for feat in self.estimators_features_
        ]
        
        # Move estimators if they support .to()
        for est in self.estimators_:
            if hasattr(est, 'to'):
                est.to(self.device)
        
        logger.info(f"Model moved to device: {self.device}")
        return self
    
    def save_model(self, filepath: str):
        """Save bagging model to file."""
        if not self.is_fitted:
            logger.warning("Saving unfitted model")
        
        # Save estimators
        estimators_state = []
        for est in self.estimators_:
            if hasattr(est, 'save_model'):
                est_state = {
                    'type': est.__class__.__name__,
                    'params': est.get_params() if hasattr(est, 'get_params') else {},
                    'state': est.__dict__.copy()
                }
            else:
                est_state = {
                    'type': est.__class__.__name__,
                    'state': est.__dict__.copy()
                }
            estimators_state.append(est_state)
        
        # Convert feature indices to CPU for saving
        features_cpu = [feat.cpu() if isinstance(feat, torch.Tensor) else feat 
                       for feat in self.estimators_features_]
        
        state = {
            'model_class': self.__class__.__name__,
            'params': self.get_params(),
            'estimators_state': estimators_state,
            'estimators_features': features_cpu,
            'is_fitted': self.is_fitted,
            'base_estimator_class': self.base_estimator.__class__.__name__ if self.base_estimator else None
        }
        
        try:
            torch.save(state, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """Load bagging model from file."""
        try:
            state = torch.load(filepath, map_location=self.device, weights_only=False)
            logger.info(f"Loading model from {filepath}")
            
            # Restore parameters
            params = state.get('params', {})
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Restore estimators
            self.estimators_ = []
            estimators_state = state.get('estimators_state', [])
            
            for est_state in estimators_state:
                est = self._make_estimator()
                
                if 'state' in est_state:
                    for key, value in est_state['state'].items():
                        if hasattr(est, key):
                            setattr(est, key, value)
                
                self.estimators_.append(est)
            
            # Restore feature indices
            features_list = state.get('estimators_features', [])
            self.estimators_features_ = [
                feat.to(self.device) if isinstance(feat, torch.Tensor) else torch.tensor(feat, device=self.device)
                for feat in features_list
            ]
            
            self.is_fitted = state.get('is_fitted', False)
            
            logger.info(f"Model loaded. n_estimators={len(self.estimators_)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        return self


class BaggingClassifier(BaggingBase):
    """
    Bagging Classifier with optimized GPU support.
    
    Implements Bootstrap Aggregating for classification tasks.
    Uses vectorized operations for fast prediction.
    
    Example:
        >>> bagging = BaggingClassifier(
        ...     base_estimator=DecisionTreeClassifier(max_depth=5),
        ...     n_estimators=50,
        ...     random_state=42
        ... )
        >>> bagging.fit(X_train, y_train)
        >>> predictions = bagging.predict(X_test)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes_ = None
        self.n_classes_ = None
    
    def _make_estimator(self):
        """Create default decision tree classifier if not provided."""
        if self.base_estimator is None:
            from .tree import DecisionTreeClassifier
            return DecisionTreeClassifier(device=self.device)
        return copy.deepcopy(self.base_estimator)
    
    def fit(self, X: Union[torch.Tensor, np.ndarray], 
            y: Union[torch.Tensor, np.ndarray],
            verbose: bool = False):
        """Fit bagging classifier."""
        y_tensor = self._to_tensor(y, dtype=torch.long)
        
        # Extract unique classes
        self.classes_ = torch.unique(y_tensor, sorted=True)
        self.n_classes_ = len(self.classes_)
        
        logger.info(f"Found {self.n_classes_} classes: {self.classes_.tolist()}")
        
        return super().fit(X, y_tensor, verbose=verbose)
    
    def predict(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Predict by majority voting from all estimators.
        OPTIMIZED: Uses vectorized operations.
        
        Args:
            X: Input features
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        n_samples = X.shape[0]
        
        # Preallocate predictions tensor
        predictions = torch.zeros((self.n_estimators, n_samples), 
                                 device=self.device, dtype=torch.long)
        
        for i, (estimator, feature_indices) in enumerate(zip(self.estimators_, 
                                                             self.estimators_features_)):
            X_subset = X[:, feature_indices]
            pred = estimator.predict(X_subset)
            
            if not isinstance(pred, torch.Tensor):
                pred = self._to_tensor(pred, dtype=torch.long)
            
            predictions[i] = pred
        
        # Majority voting (mode along estimators dimension)
        final_predictions = torch.mode(predictions, dim=0)[0]
        
        return self._to_numpy(final_predictions)
    
    def predict_proba(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        OPTIMIZED: Uses vectorized vote counting.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        n_samples = X.shape[0]
        
        # Accumulate votes for each class
        votes = torch.zeros((n_samples, self.n_classes_), 
                           device=self.device, dtype=torch.float32)
        
        for estimator, feature_indices in zip(self.estimators_, 
                                              self.estimators_features_):
            X_subset = X[:, feature_indices]
            pred = estimator.predict(X_subset)
            
            if not isinstance(pred, torch.Tensor):
                pred = self._to_tensor(pred, dtype=torch.long)
            
            # One-hot encode and add to votes (vectorized)
            for i, cls in enumerate(self.classes_):
                votes[:, i] += (pred == cls).float()
        
        # Normalize to probabilities
        probabilities = votes / self.n_estimators
        
        return self._to_numpy(probabilities)
    
    def score(self, X: Union[torch.Tensor, np.ndarray], 
              y: Union[torch.Tensor, np.ndarray]) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        return np.mean(predictions == y)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params(deep=deep)
        params.update({
            'n_classes_': self.n_classes_,
            'classes_': self.classes_.tolist() if self.classes_ is not None else None
        })
        return params
    
    
# if __name__ == '__main__':
#     import numpy as np
#     import torch
#     from sklearn.datasets import load_iris
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score, r2_score

#     # Load dataset
#     iris = load_iris()
#     X = iris.data
#     y_class = iris.target
#     y_reg = iris.data[:, 0]  # Use sepal length as regression target for demo

#     # Classification split
#     X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_class, test_size=0.3, random_state=42)

#     # Regression split
#     X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)

#     device = torch.device('cuda')  # or 'cuda' if using GPU

#     # ========== CLASSIFICATION ==========
#     print("=== Classification ===")

#     # BaggingClassifier
#     bag_clf = BaggingClassifier(
#         base_estimator=DecisionTreeClassifier(max_depth=3),
#         n_estimators=25,
#         random_state=42,
#         device=device
#     )
#     bag_clf.fit(X_train_clf, y_train_clf)
#     y_pred_bag = bag_clf.predict(X_test_clf)
#     acc_bag = accuracy_score(y_test_clf, y_pred_bag)
#     print(f"[BaggingClassifier] Accuracy: {acc_bag:.4f}")

#     # Sklearn RandomForestClassifier
#     rf_clf = RandomForestClassifier(n_estimators=25, max_depth=3, random_state=42)
#     rf_clf.fit(X_train_clf, y_train_clf)
#     y_pred_rf = rf_clf.predict(X_test_clf)
#     acc_rf = accuracy_score(y_test_clf, y_pred_rf)
#     print(f"[Sklearn RF Classifier] Accuracy: {acc_rf:.4f}")

#     # ========== REGRESSION ==========
#     print("\n=== Regression ===")

#     # BaggingRegressor
#     bag_reg = BaggingRegressor(
#         base_estimator=DecisionTreeRegressor(max_depth=3),
#         n_estimators=25,
#         random_state=42,
#         device=device
#     )
#     bag_reg.fit(X_train_reg, y_train_reg)
#     y_pred_bag_reg = bag_reg.predict(X_test_reg)
#     r2_bag = r2_score(y_test_reg, y_pred_bag_reg)
#     print(f"[BaggingRegressor] R² Score: {r2_bag:.4f}")

#     # HybridRandomForestRegressor
#     hybrid_rf = RandomForestRegressor(
#         n_estimators=25,
#         max_depth=3,
#         random_state=42,
#         device=device
#     )
#     hybrid_rf.fit(X_train_reg, y_train_reg)
#     y_pred_hybrid_rf = hybrid_rf.predict(X_test_reg)
#     r2_hybrid = r2_score(y_test_reg, y_pred_hybrid_rf)
#     print(f"[HybridRandomForestRegressor] R² Score: {r2_hybrid:.4f}")

#     # ========== SUMMARY ==========
#     print("\n=== Summary ===")
#     print(f"BaggingClassifier Accuracy          : {acc_bag:.4f}")
#     print(f"RandomForestClassifier Accuracy     : {acc_rf:.4f}")
#     print(f"BaggingRegressor R² Score           : {r2_bag:.4f}")
#     print(f"HybridRandomForestRegressor R² Score: {r2_hybrid:.4f}")
