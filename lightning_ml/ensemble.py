"""
Random Forest Implementation using PyTorch Decision Trees
Ensemble method for both classification and regression
"""

import torch
import numpy as np
from typing import Union, Optional
from concurrent.futures import ThreadPoolExecutor
from .base_model import BaseTreeModel, BaseSupervisedModel
from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestBase(BaseTreeModel):
    """
    Base class for Random Forest with parallel tree training.
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str]] = 'sqrt',
                 bootstrap: bool = True,
                 max_samples: Optional[float] = None,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize Random Forest.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features per tree
            bootstrap: Whether to use bootstrap samples
            max_samples: Maximum samples per tree (if bootstrap=True)
            n_jobs: Number of parallel jobs (-1 = all cores)
            random_state: Random seed
            device: Computation device
        """
        super().__init__(max_depth, min_samples_split, min_samples_leaf, device)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.random_state = random_state
        
        self.estimators_ = []
        self.n_features_ = None
        self.n_outputs_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
    
    def _create_tree(self):
        """Create a single decision tree (to be overridden)."""
        raise NotImplementedError
    
    def _bootstrap_sample(self, X: torch.Tensor, y: torch.Tensor, 
                         tree_idx: int) -> tuple:
        """
        Create bootstrap sample for a tree.
        
        Args:
            X: Feature tensor
            y: Target tensor
            tree_idx: Tree index (for reproducibility)
            
        Returns:
            Tuple of (X_sample, y_sample)
        """
        n_samples = len(X)
        
        if self.bootstrap:
            # Determine sample size
            if self.max_samples is None:
                sample_size = n_samples
            elif isinstance(self.max_samples, float):
                sample_size = int(self.max_samples * n_samples)
            else:
                sample_size = self.max_samples
            
            # Bootstrap sampling with replacement
            indices = torch.randint(0, n_samples, (sample_size,), device=self.device)
        else:
            # Use all samples
            indices = torch.arange(n_samples, device=self.device)
        
        return X[indices], y[indices]
    
    def _fit_tree(self, tree_idx: int, X: torch.Tensor, y: torch.Tensor):
        """
        Fit a single tree.
        
        Args:
            tree_idx: Tree index
            X: Feature tensor
            y: Target tensor
            
        Returns:
            Fitted tree
        """
        # Create tree
        tree = self._create_tree()
        
        # Bootstrap sample
        X_sample, y_sample = self._bootstrap_sample(X, y, tree_idx)
        
        # Fit tree
        tree.fit(X_sample, y_sample, verbose=False)
        
        return tree
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'RandomForestBase':
        """
        Fit random forest.
        
        Args:
            X: Training features
            y: Training targets
            verbose: Print progress
            
        Returns:
            self: Fitted model
        """
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y, dtype=torch.float32)
        
        self._validate_input(X, y)
        
        self.n_features_ = X.shape[1]
        
        # Train trees in parallel
        if self.n_jobs is not None and self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(self._fit_tree, i, X, y)
                    for i in range(self.n_estimators)
                ]
                self.estimators_ = [f.result() for f in futures]
        else:
            # Sequential training
            self.estimators_ = []
            for i in range(self.n_estimators):
                tree = self._fit_tree(i, X, y)
                self.estimators_.append(tree)
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"Trained {i+1}/{self.n_estimators} trees")
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        self._update_fit_info(X)
        
        if verbose:
            print(f"Random Forest fitted with {self.n_estimators} trees")
        
        return self
    
    def _calculate_feature_importances(self):
        """Aggregate feature importances from all trees."""
        if not self.estimators_:
            return
        
        self.feature_importances_ = np.zeros(self.n_features_)
        
        for tree in self.estimators_:
            if hasattr(tree, 'feature_importances_'):
                self.feature_importances_ += tree.feature_importances_
        
        # Normalize
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.n_estimators
    
    def get_params(self):
        params = super().get_params()
        params.update({
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'max_samples': self.max_samples,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state
        })
        return params


class RandomForestRegressor(RandomForestBase):
    """
    Random Forest Regressor with GPU support.
    
    Args:
        n_estimators: Number of trees in forest
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        max_features: Features to consider ('sqrt', 'log2', int, None)
        bootstrap: Use bootstrap sampling
        max_samples: Maximum samples per tree
        n_jobs: Parallel jobs
        random_state: Random seed
        device: Computation device
    """
    
    def _create_tree(self):
        """Create decision tree regressor."""
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            device=self.device
        )
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict by averaging predictions from all trees.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        
        # Average predictions
        return predictions.mean(axis=0)
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate R² score."""
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


class RandomForestClassifier(RandomForestBase):
    """
    Random Forest Classifier with GPU support.
    
    Args:
        n_estimators: Number of trees in forest
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        max_features: Features to consider ('sqrt', 'log2', int, None)
        criterion: Split criterion ('gini' or 'entropy')
        bootstrap: Use bootstrap sampling
        max_samples: Maximum samples per tree
        n_jobs: Parallel jobs
        random_state: Random seed
        device: Computation device
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str]] = 'sqrt',
                 criterion: str = 'gini',
                 bootstrap: bool = True,
                 max_samples: Optional[float] = None,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        super().__init__(
            n_estimators, max_depth, min_samples_split, min_samples_leaf,
            max_features, bootstrap, max_samples, n_jobs, random_state, device
        )
        self.criterion = criterion
        self.classes_ = None
        self.n_classes_ = None
    
    def _create_tree(self):
        """Create decision tree classifier."""
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=self.criterion,
            device=self.device
        )
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'RandomForestClassifier':
        """Fit random forest classifier."""
        # Get unique classes
        y_np = y if isinstance(y, np.ndarray) else self._to_numpy(y)
        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        
        # Convert to long tensor
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y_np, dtype=torch.long)
        
        # Call parent fit
        return super().fit(X, y, verbose)
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities by averaging tree predictions.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        n_samples = len(X)
        
        # Collect probability predictions from all trees
        all_probas = np.zeros((self.n_estimators, n_samples, self.n_classes_))
        
        for i, tree in enumerate(self.estimators_):
            all_probas[i] = tree.predict_proba(X)
        
        # Average probabilities
        return all_probas.mean(axis=0)
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return self.classes_[predictions]
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        return np.mean(predictions == y)
    
    def get_params(self):
        params = super().get_params()
        params['criterion'] = self.criterion
        return params

"""
Bagging Classifier and Regressor Implementation using PyTorch
Bootstrap Aggregating for any base estimator
"""

import torch
import numpy as np
from typing import Union, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from .base_model import BaseTreeModel, BaseSupervisedModel
import copy


class BaggingBase(BaseSupervisedModel):
    """
    Base class for Bagging with parallel estimator training.
    """
    
    def __init__(self,
                 base_estimator: Optional[Any] = None,
                 n_estimators: int = 10,
                 max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize Bagging ensemble.
        
        Args:
            base_estimator: Base estimator to use (default: DecisionTreeClassifier/Regressor)
            n_estimators: Number of estimators
            max_samples: Number/fraction of samples per estimator
            max_features: Number/fraction of features per estimator
            bootstrap: Whether to use bootstrap samples
            bootstrap_features: Whether to bootstrap features
            n_jobs: Number of parallel jobs (-1 = all cores)
            random_state: Random seed
            device: Computation device
        """
        super().__init__(device)
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.random_state = random_state
        
        self.estimators_ = []
        self.estimators_features_ = []
        self.n_features_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
    
    def _make_estimator(self):
        """Create a copy of base estimator."""
        if self.base_estimator is None:
            # Default estimator will be set by child classes
            raise NotImplementedError("Must provide base_estimator or override _make_estimator")
        
        estimator = copy.deepcopy(self.base_estimator)
        return estimator
    
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
    
    def _generate_sample_indices(self, n_samples: int, n_samples_draw: int) -> torch.Tensor:
        """Generate sample indices (with or without replacement)."""
        if self.bootstrap:
            indices = torch.randint(0, n_samples, (n_samples_draw,), device=self.device)
        else:
            indices = torch.randperm(n_samples, device=self.device)[:n_samples_draw]
        return indices
    
    def _generate_feature_indices(self, n_features: int, n_features_draw: int) -> np.ndarray:
        """Generate feature indices (with or without replacement)."""
        if self.bootstrap_features:
            indices = np.random.randint(0, n_features, n_features_draw)
        else:
            indices = np.random.permutation(n_features)[:n_features_draw]
        return np.sort(indices)
    
    def _fit_estimator(self, estimator_idx: int, X: torch.Tensor, y: torch.Tensor):
        """
        Fit a single estimator.
        
        Args:
            estimator_idx: Estimator index
            X: Feature tensor
            y: Target tensor
            
        Returns:
            Tuple of (fitted estimator, feature indices used)
        """
        n_samples = len(X)
        n_features = X.shape[1]
        
        # Create estimator
        estimator = self._make_estimator()
        
        # Sample indices
        n_samples_draw = self._get_n_samples(n_samples)
        sample_indices = self._generate_sample_indices(n_samples, n_samples_draw)
        
        # Feature indices
        n_features_draw = self._get_n_features(n_features)
        feature_indices = self._generate_feature_indices(n_features, n_features_draw)
        
        # Extract samples and features
        X_subset = X[sample_indices][:, feature_indices]
        y_subset = y[sample_indices]
        
        # Fit estimator
        estimator.fit(X_subset, y_subset)
        
        return estimator, feature_indices
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'BaggingBase':
        """
        Fit bagging ensemble.
        
        Args:
            X: Training features
            y: Training targets
            verbose: Print progress
            
        Returns:
            self: Fitted model
        """
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y, dtype=torch.float32)
        
        self._validate_input(X, y)
        
        self.n_features_ = X.shape[1]
        
        # Train estimators in parallel
        if self.n_jobs is not None and self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(self._fit_estimator, i, X, y)
                    for i in range(self.n_estimators)
                ]
                results = [f.result() for f in futures]
        else:
            # Sequential training
            results = []
            for i in range(self.n_estimators):
                result = self._fit_estimator(i, X, y)
                results.append(result)
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"Trained {i+1}/{self.n_estimators} estimators")
        
        # Unpack results
        self.estimators_ = [r[0] for r in results]
        self.estimators_features_ = [r[1] for r in results]
        
        self._update_fit_info(X)
        
        if verbose:
            print(f"Bagging ensemble fitted with {self.n_estimators} estimators")
        
        return self
    
    def get_params(self):
        params = super().get_params()
        params.update({
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state
        })
        return params


class BaggingRegressor(BaggingBase):
    """
    Bagging Regressor with GPU support.
    
    Implements Bootstrap Aggregating for regression tasks.
    
    Args:
        base_estimator: Base regression estimator (default: DecisionTreeRegressor)
        n_estimators: Number of estimators
        max_samples: Number/fraction of samples per estimator (1.0 = 100%)
        max_features: Number/fraction of features per estimator (1.0 = 100%)
        bootstrap: Use bootstrap sampling (with replacement)
        bootstrap_features: Use bootstrap feature sampling
        n_jobs: Parallel jobs (-1 = all cores)
        random_state: Random seed
        device: Computation device
    
    Example:
        >>> from lightning_ml.tree import DecisionTreeRegressor
        >>> bagging = BaggingRegressor(
        ...     base_estimator=DecisionTreeRegressor(max_depth=5),
        ...     n_estimators=50,
        ...     random_state=42
        ... )
        >>> bagging.fit(X_train, y_train)
        >>> predictions = bagging.predict(X_test)
    """
    
    def _make_estimator(self):
        """Create default decision tree regressor if not provided."""
        if self.base_estimator is None:
            from .tree import DecisionTreeRegressor
            return DecisionTreeRegressor(device=self.device)
        return copy.deepcopy(self.base_estimator)
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict by averaging predictions from all estimators.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        
        # Collect predictions from all estimators
        predictions = []
        for estimator, feature_indices in zip(self.estimators_, self.estimators_features_):
            X_subset = X[:, feature_indices]
            pred = estimator.predict(X_subset)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Average predictions
        return predictions.mean(axis=0)
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate R² score.
        
        Args:
            X: Test features
            y: True values
            
        Returns:
            R² score
        """
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


class BaggingClassifier(BaggingBase):
    """
    Bagging Classifier with GPU support.
    
    Implements Bootstrap Aggregating for classification tasks.
    
    Args:
        base_estimator: Base classification estimator (default: DecisionTreeClassifier)
        n_estimators: Number of estimators
        max_samples: Number/fraction of samples per estimator (1.0 = 100%)
        max_features: Number/fraction of features per estimator (1.0 = 100%)
        bootstrap: Use bootstrap sampling (with replacement)
        bootstrap_features: Use bootstrap feature sampling
        n_jobs: Parallel jobs (-1 = all cores)
        random_state: Random seed
        device: Computation device
    
    Example:
        >>> from lightning_ml.tree import DecisionTreeClassifier
        >>> bagging = BaggingClassifier(
        ...     base_estimator=DecisionTreeClassifier(max_depth=5),
        ...     n_estimators=50,
        ...     random_state=42
        ... )
        >>> bagging.fit(X_train, y_train)
        >>> predictions = bagging.predict(X_test)
        >>> probabilities = bagging.predict_proba(X_test)
    """
    
    def __init__(self,
                 base_estimator: Optional[Any] = None,
                 n_estimators: int = 10,
                 max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        super().__init__(
            base_estimator, n_estimators, max_samples, max_features,
            bootstrap, bootstrap_features, n_jobs, random_state, device
        )
        self.classes_ = None
        self.n_classes_ = None
    
    def _make_estimator(self):
        """Create default decision tree classifier if not provided."""
        if self.base_estimator is None:
            from .tree import DecisionTreeClassifier
            return DecisionTreeClassifier(device=self.device)
        return copy.deepcopy(self.base_estimator)
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'BaggingClassifier':
        """
        Fit bagging classifier.
        
        Args:
            X: Training features
            y: Training targets
            verbose: Print progress
            
        Returns:
            self: Fitted model
        """
        # Get unique classes
        y_np = y if isinstance(y, np.ndarray) else self._to_numpy(y)
        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        
        # Convert to long tensor
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y_np, dtype=torch.long)
        
        # Call parent fit
        return super().fit(X, y, verbose)
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities by averaging estimator predictions.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        n_samples = len(X)
        
        # Collect probability predictions from all estimators
        all_probas = np.zeros((self.n_estimators, n_samples, self.n_classes_))
        
        for i, (estimator, feature_indices) in enumerate(zip(self.estimators_, 
                                                              self.estimators_features_)):
            X_subset = X[:, feature_indices]
            
            # Check if estimator has predict_proba
            if hasattr(estimator, 'predict_proba'):
                all_probas[i] = estimator.predict_proba(X_subset)
            else:
                # Fall back to hard predictions
                predictions = estimator.predict(X_subset)
                for j, pred in enumerate(predictions):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    all_probas[i, j, class_idx] = 1.0
        
        # Average probabilities
        return all_probas.mean(axis=0)
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return self.classes_[predictions]
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        return np.mean(predictions == y)