"""
K-Nearest Neighbors (KNN) and K-Means Clustering
PyTorch implementations with GPU acceleration
"""

import torch
import numpy as np
from typing import Union, Optional, Literal
from .base_model import BaseSupervisedModel, BaseUnsupervisedModel


# ============================================================================
# K-NEAREST NEIGHBORS (KNN)
# ============================================================================

class KNNBase(BaseSupervisedModel):
    """Base class for KNN algorithms."""
    
    def __init__(self,
                 n_neighbors: int = 5,
                 weights: Literal['uniform', 'distance'] = 'uniform',
                 metric: str = 'euclidean',
                 device: Optional[torch.device] = None):
        """
        Initialize KNN base.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform' or 'distance')
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            device: Computation device
        """
        super().__init__(device)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def _calculate_distances(self, X: torch.Tensor, X_train: torch.Tensor) -> torch.Tensor:
        """
        Calculate pairwise distances between X and X_train.
        
        Args:
            X: Query points (n_samples, n_features)
            X_train: Training points (n_train, n_features)
            
        Returns:
            Distance matrix (n_samples, n_train)
        """
        if self.metric == 'euclidean':
            # Efficient euclidean distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            x_norm = (X ** 2).sum(1).view(-1, 1)
            y_norm = (X_train ** 2).sum(1).view(1, -1)
            distances = x_norm + y_norm - 2.0 * torch.mm(X, X_train.T)
            distances = torch.sqrt(torch.clamp(distances, min=0.0))
            
        elif self.metric == 'manhattan':
            # Manhattan distance using broadcasting
            distances = torch.cdist(X, X_train, p=1)
            
        elif self.metric == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
            X_train_norm = X_train / (X_train.norm(dim=1, keepdim=True) + 1e-8)
            similarities = torch.mm(X_norm, X_train_norm.T)
            distances = 1 - similarities
            
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return distances
    
    def _get_neighbors(self, distances: torch.Tensor) -> tuple:
        """
        Get k nearest neighbors.
        
        Args:
            distances: Distance matrix
            
        Returns:
            Tuple of (neighbor_distances, neighbor_indices)
        """
        k = min(self.n_neighbors, distances.shape[1])
        neighbor_distances, neighbor_indices = torch.topk(
            distances, k, largest=False, dim=1
        )
        return neighbor_distances, neighbor_indices
    
    def _get_weights(self, distances: torch.Tensor) -> torch.Tensor:
        """Calculate weights for neighbors."""
        if self.weights == 'uniform':
            return torch.ones_like(distances)
        else:  # distance
            # Avoid division by zero
            weights = 1.0 / (distances + 1e-8)
            return weights
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'KNNBase':
        """Store training data."""
        self.X_train = self._to_tensor(X, dtype=torch.float32)
        self.y_train = self._to_tensor(y, dtype=torch.float32)
        
        self._validate_input(self.X_train, self.y_train)
        self._update_fit_info(self.X_train)
        
        if verbose:
            print(f"KNN fitted with {len(self.X_train)} training samples")
        
        return self
    
    def get_params(self):
        params = super().get_params()
        params.update({
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric
        })
        return params


class KNNClassifier(KNNBase):
    """
    K-Nearest Neighbors Classifier with GPU support.
    
    Args:
        n_neighbors: Number of neighbors
        weights: Weight function ('uniform' or 'distance')
        metric: Distance metric
        device: Computation device
    """
    
    def __init__(self,
                 n_neighbors: int = 5,
                 weights: Literal['uniform', 'distance'] = 'uniform',
                 metric: str = 'euclidean',
                 device: Optional[torch.device] = None):
        super().__init__(n_neighbors, weights, metric, device)
        self.classes_ = None
        self.n_classes_ = None
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'KNNClassifier':
        """Fit KNN classifier."""
        # Get unique classes
        y_np = y if isinstance(y, np.ndarray) else self._to_numpy(y)
        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        
        # Store as long tensor for classification
        self.X_train = self._to_tensor(X, dtype=torch.float32)
        self.y_train = self._to_tensor(y_np, dtype=torch.long)
        
        self._validate_input(self.X_train, self.y_train)
        self._update_fit_info(self.X_train)
        
        if verbose:
            print(f"KNN Classifier fitted: {len(self.X_train)} samples, {self.n_classes_} classes")
        
        return self
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        
        # Calculate distances
        distances = self._calculate_distances(X, self.X_train)
        neighbor_distances, neighbor_indices = self._get_neighbors(distances)
        
        # Get neighbor labels
        neighbor_labels = self.y_train[neighbor_indices]
        
        # Calculate weights
        weights = self._get_weights(neighbor_distances)
        
        # Calculate weighted votes for each class
        n_samples = len(X)
        probas = torch.zeros(n_samples, self.n_classes_, device=self.device)
        
        for i in range(n_samples):
            for j, label in enumerate(neighbor_labels[i]):
                probas[i, label] += weights[i, j]
        
        # Normalize to probabilities
        probas = probas / probas.sum(dim=1, keepdim=True)
        
        return self._to_numpy(probas)
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return self.classes_[predictions]
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate accuracy."""
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        return np.mean(predictions == y)


class KNNRegressor(KNNBase):
    """
    K-Nearest Neighbors Regressor with GPU support.
    
    Args:
        n_neighbors: Number of neighbors
        weights: Weight function ('uniform' or 'distance')
        metric: Distance metric
        device: Computation device
    """
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict continuous values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        
        # Calculate distances
        distances = self._calculate_distances(X, self.X_train)
        neighbor_distances, neighbor_indices = self._get_neighbors(distances)
        
        # Get neighbor values
        neighbor_values = self.y_train[neighbor_indices]
        
        # Calculate weights
        weights = self._get_weights(neighbor_distances)
        
        # Weighted average
        weighted_sum = (neighbor_values * weights).sum(dim=1)
        weight_sum = weights.sum(dim=1)
        predictions = weighted_sum / weight_sum
        
        return self._to_numpy(predictions)
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate R² score."""
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


