"""
K-Nearest Neighbors (KNN) with Save/Load/To Methods
PyTorch implementations with GPU acceleration and persistence
"""

import torch
import numpy as np
from typing import Union, Optional, Literal
from .base_model import BaseSupervisedModel, BaseUnsupervisedModel
import logging

logger = logging.getLogger(__name__)


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
        """Calculate pairwise distances between X and X_train."""
        if self.metric == 'euclidean':
            x_norm = (X ** 2).sum(1).view(-1, 1)
            y_norm = (X_train ** 2).sum(1).view(1, -1)
            distances = x_norm + y_norm - 2.0 * torch.mm(X, X_train.T)
            distances = torch.sqrt(torch.clamp(distances, min=0.0))
            
        elif self.metric == 'manhattan':
            distances = torch.cdist(X, X_train, p=1)
            
        elif self.metric == 'cosine':
            X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
            X_train_norm = X_train / (X_train.norm(dim=1, keepdim=True) + 1e-8)
            similarities = torch.mm(X_norm, X_train_norm.T)
            distances = 1 - similarities
            
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return distances
    
    def _get_neighbors(self, distances: torch.Tensor) -> tuple:
        """Get k nearest neighbors."""
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
    
    def get_params(self, deep: bool = True):
        params = super().get_params(deep=deep)
        params.update({
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric
        })
        return params
    
    def save(self, filepath: str):
        """
        Save KNN model to file.
        
        Args:
            filepath: Path to save model
        
        Example:
            >>> knn.save('knn_model.pt')
        """
        if not self.is_fitted:
            import warnings
            warnings.warn("Saving unfitted KNN model")
        
        state = {
            'model_class': self.__class__.__name__,
            'params': self.get_params(),
            'is_fitted': self.is_fitted,
            'X_train': self.X_train,
            'y_train': self.y_train,
            'n_features_in': self.n_features_in_,
            'n_samples_seen': self.n_samples_seen_,
            'training_history': self._training_history
        }
        
        try:
            torch.save(state, filepath)
            logger.info(f"KNN model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save KNN model to {filepath}: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """
        Load KNN model from file.
        
        Args:
            filepath: Path to model file
        
        Example:
            >>> knn = KNNClassifier()
            >>> knn.load('knn_model.pt')
        """
        try:
            state = torch.load(filepath, map_location=self.device, weights_only=False)
            logger.info(f"Loading KNN model from {filepath}")
            
            # Restore parameters
            params = state.get('params', {})
            self.n_neighbors = params.get('n_neighbors', self.n_neighbors)
            self.weights = params.get('weights', self.weights)
            self.metric = params.get('metric', self.metric)
            
            # Restore state
            self.is_fitted = state.get('is_fitted', False)
            self.n_features_in_ = state.get('n_features_in', None)
            self.n_samples_seen_ = state.get('n_samples_seen', 0)
            self._training_history = state.get('training_history', {'loss': [], 'epoch': []})
            
            # Restore training data and move to current device
            self.X_train = state.get('X_train', None)
            self.y_train = state.get('y_train', None)
            
            if self.X_train is not None:
                self.X_train = self.X_train.to(self.device)
            if self.y_train is not None:
                self.y_train = self.y_train.to(self.device)
            
            logger.info(f"KNN model loaded successfully. is_fitted={self.is_fitted}")
            if self.X_train is not None:
                logger.info(f"  Training data: {self.X_train.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load KNN model from {filepath}: {str(e)}")
            raise
        
        return self
    
    def to(self, device: Union[str, torch.device]):
        """
        Move KNN model to specified device.
        
        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
        
        Returns:
            self: Model on new device
        
        Example:
            >>> knn.to('cuda')
            >>> knn.to(torch.device('cpu'))
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Move training data to new device
        if self.X_train is not None:
            self.X_train = self.X_train.to(self.device)
        if self.y_train is not None:
            self.y_train = self.y_train.to(self.device)
        
        logger.info(f"KNN model moved to device: {self.device}")
        return self


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
        
        distances = self._calculate_distances(X, self.X_train)
        neighbor_distances, neighbor_indices = self._get_neighbors(distances)
        neighbor_labels = self.y_train[neighbor_indices]
        weights = self._get_weights(neighbor_distances)
        
        n_samples = len(X)
        probas = torch.zeros(n_samples, self.n_classes_, device=self.device)
        
        for i in range(n_samples):
            for j, label in enumerate(neighbor_labels[i]):
                probas[i, label] += weights[i, j]
        
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
    
    def save(self, filepath: str):
        """Save KNN Classifier to file."""
        if not self.is_fitted:
            import warnings
            warnings.warn("Saving unfitted KNN Classifier")
        
        state = {
            'model_class': self.__class__.__name__,
            'params': self.get_params(),
            'is_fitted': self.is_fitted,
            'X_train': self.X_train,
            'y_train': self.y_train,
            'classes': self.classes_,
            'n_classes': self.n_classes_,
            'n_features_in': self.n_features_in_,
            'n_samples_seen': self.n_samples_seen_,
            'training_history': self._training_history
        }
        
        try:
            torch.save(state, filepath)
            logger.info(f"KNN Classifier saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save KNN Classifier to {filepath}: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """Load KNN Classifier from file."""
        try:
            state = torch.load(filepath, map_location=self.device, weights_only=False)
            logger.info(f"Loading KNN Classifier from {filepath}")
            
            # Restore parameters
            params = state.get('params', {})
            self.n_neighbors = params.get('n_neighbors', self.n_neighbors)
            self.weights = params.get('weights', self.weights)
            self.metric = params.get('metric', self.metric)
            
            # Restore classifier-specific state
            self.classes_ = state.get('classes', None)
            self.n_classes_ = state.get('n_classes', None)
            
            # Restore base state
            self.is_fitted = state.get('is_fitted', False)
            self.n_features_in_ = state.get('n_features_in', None)
            self.n_samples_seen_ = state.get('n_samples_seen', 0)
            self._training_history = state.get('training_history', {'loss': [], 'epoch': []})
            
            # Restore training data
            self.X_train = state.get('X_train', None)
            self.y_train = state.get('y_train', None)
            
            if self.X_train is not None:
                self.X_train = self.X_train.to(self.device)
            if self.y_train is not None:
                self.y_train = self.y_train.to(self.device)
            
            logger.info(f"KNN Classifier loaded successfully. is_fitted={self.is_fitted}")
            logger.info(f"  Classes: {self.n_classes_}, Samples: {self.n_samples_seen_}")
            
        except Exception as e:
            logger.error(f"Failed to load KNN Classifier from {filepath}: {str(e)}")
            raise
        
        return self


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
        
        distances = self._calculate_distances(X, self.X_train)
        neighbor_distances, neighbor_indices = self._get_neighbors(distances)
        neighbor_values = self.y_train[neighbor_indices]
        weights = self._get_weights(neighbor_distances)
        
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
    
    
# """
# Test script for KNN save/load/to methods
# """

# import numpy as np
# import torch
# import os
# from sklearn.datasets import make_classification, make_regression
# from sklearn.model_selection import train_test_split


# def create_classification_data():
#     """Create sample classification dataset."""
#     X, y = make_classification(
#         n_samples=200,
#         n_features=10,
#         n_informative=5,
#         n_redundant=2,
#         n_classes=3,
#         random_state=42
#     )
#     return train_test_split(X, y, test_size=0.3, random_state=42)


# def create_regression_data():
#     """Create sample regression dataset."""
#     X, y = make_regression(
#         n_samples=200,
#         n_features=10,
#         noise=0.1,
#         random_state=42
#     )
#     return train_test_split(X, y, test_size=0.3, random_state=42)


# def test_knn_classifier_save_load():
#     """Test KNNClassifier save and load functionality."""
#     print("=" * 60)
#     print("TEST 1: KNNClassifier Save and Load")
#     print("=" * 60)
    
#     # Import KNN
#     # from lightning_ml import KNNClassifier  # Uncomment in actual use
#       # Assuming file is named knn.py
    
#     # Create data
#     print("\n1. Creating classification dataset...")
#     X_train, X_test, y_train, y_test = create_classification_data()
#     print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
#     print(f"   Classes: {np.unique(y_train)}")
    
#     # Train original model
#     print("\n2. Training original KNN Classifier...")
#     knn1 = KNNClassifier(n_neighbors=5, weights='distance', metric='euclidean')
#     knn1.fit(X_train, y_train, verbose=True)
    
#     # Evaluate original
#     score1 = knn1.score(X_test, y_test)
#     pred1 = knn1.predict(X_test[:5])
#     print(f"   Original model accuracy: {score1:.4f}")
#     print(f"   Sample predictions: {pred1}")
    
#     # Save model
#     filepath = "knn_classifier_test.pt"
#     print(f"\n3. Saving model to {filepath}...")
#     knn1.save(filepath)
#     print("   ✓ Model saved successfully")
    
#     # Load model
#     print("\n4. Loading model into new instance...")
#     knn2 = KNNClassifier()
#     knn2.load(filepath)
#     print("   ✓ Model loaded successfully")
    
#     # Evaluate loaded model
#     score2 = knn2.score(X_test, y_test)
#     pred2 = knn2.predict(X_test[:5])
#     print(f"   Loaded model accuracy: {score2:.4f}")
#     print(f"   Sample predictions: {pred2}")
    
#     # Compare
#     print("\n5. Comparing models:")
#     print(f"   Scores match: {score1 == score2}")
#     print(f"   Predictions match: {np.array_equal(pred1, pred2)}")
#     print(f"   n_neighbors match: {knn1.n_neighbors == knn2.n_neighbors}")
#     print(f"   Training data shape match: {knn1.X_train.shape == knn2.X_train.shape}")
    
#     # Cleanup
#     if os.path.exists(filepath):
#         os.remove(filepath)
#         print(f"\n6. Cleaned up: {filepath}")
    
#     print("\n✓ TEST 1 PASSED\n")


# def test_knn_regressor_save_load():
#     """Test KNNRegressor save and load functionality."""
#     print("=" * 60)
#     print("TEST 2: KNNRegressor Save and Load")
#     print("=" * 60)
    
    
    
#     # Create data
#     print("\n1. Creating regression dataset...")
#     X_train, X_test, y_train, y_test = create_regression_data()
#     print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
#     # Train original model
#     print("\n2. Training original KNN Regressor...")
#     knn1 = KNNRegressor(n_neighbors=3, weights='uniform', metric='euclidean')
#     knn1.fit(X_train, y_train, verbose=True)
    
#     # Evaluate original
#     score1 = knn1.score(X_test, y_test)
#     pred1 = knn1.predict(X_test[:5])
#     print(f"   Original model R² score: {score1:.4f}")
#     print(f"   Sample predictions: {pred1}")
    
#     # Save model
#     filepath = "knn_regressor_test.pt"
#     print(f"\n3. Saving model to {filepath}...")
#     knn1.save(filepath)
#     print("   ✓ Model saved successfully")
    
#     # Load model
#     print("\n4. Loading model into new instance...")
#     knn2 = KNNRegressor()
#     knn2.load(filepath)
#     print("   ✓ Model loaded successfully")
    
#     # Evaluate loaded model
#     score2 = knn2.score(X_test, y_test)
#     pred2 = knn2.predict(X_test[:5])
#     print(f"   Loaded model R² score: {score2:.4f}")
#     print(f"   Sample predictions: {pred2}")
    
#     # Compare
#     print("\n5. Comparing models:")
#     print(f"   Scores match: {abs(score1 - score2) < 1e-6}")
#     print(f"   Predictions match: {np.allclose(pred1, pred2)}")
#     print(f"   Parameters match: {knn1.get_params() == knn2.get_params()}")
    
#     # Cleanup
#     if os.path.exists(filepath):
#         os.remove(filepath)
#         print(f"\n6. Cleaned up: {filepath}")
    
#     print("\n✓ TEST 2 PASSED\n")


# def test_device_movement():
#     """Test device movement (to method)."""
#     print("=" * 60)
#     print("TEST 3: Device Movement (to method)")
#     print("=" * 60)
    
    
    
#     # Create data
#     print("\n1. Creating dataset...")
#     X_train, X_test, y_train, y_test = create_classification_data()
    
#     # Create and fit model
#     print("\n2. Creating and fitting KNN Classifier...")
#     knn = KNNClassifier(n_neighbors=5)
#     print(f"   Initial device: {knn.device}")
    
#     knn.fit(X_train, y_train, verbose=False)
#     score_initial = knn.score(X_test, y_test)
#     print(f"   Initial accuracy: {score_initial:.4f}")
    
#     # Move to CPU
#     print("\n3. Moving to CPU...")
#     knn.to('cpu')
#     print(f"   Current device: {knn.device}")
#     print(f"   X_train device: {knn.X_train.device}")
#     print(f"   y_train device: {knn.y_train.device}")
    
#     score_cpu = knn.score(X_test, y_test)
#     print(f"   Accuracy after move: {score_cpu:.4f}")
#     print(f"   Scores match: {score_initial == score_cpu}")
    
#     # Test CUDA if available
#     if torch.cuda.is_available():
#         print("\n4. Moving to CUDA...")
#         knn.to('cuda')
#         print(f"   Current device: {knn.device}")
#         print(f"   X_train device: {knn.X_train.device}")
        
#         score_cuda = knn.score(X_test, y_test)
#         print(f"   Accuracy on CUDA: {score_cuda:.4f}")
#         print(f"   Scores match: {score_initial == score_cuda}")
        
#         # Move back to CPU
#         knn.to(torch.device('cpu'))
#         print(f"   Moved back to: {knn.device}")
#     else:
#         print("\n4. CUDA not available, skipping CUDA test")
    
#     print("\n✓ TEST 3 PASSED\n")


# def test_save_load_with_device_change():
#     """Test saving on one device and loading on another."""
#     print("=" * 60)
#     print("TEST 4: Save/Load with Device Change")
#     print("=" * 60)
    
    
    
#     # Create data
#     print("\n1. Creating dataset...")
#     X_train, X_test, y_train, y_test = create_classification_data()
    
#     # Train on CPU
#     print("\n2. Training on CPU...")
#     knn1 = KNNClassifier(n_neighbors=5, device=torch.device('cpu'))
#     knn1.fit(X_train, y_train, verbose=False)
#     score1 = knn1.score(X_test, y_test)
#     print(f"   Device: {knn1.device}")
#     print(f"   Accuracy: {score1:.4f}")
    
#     # Save
#     filepath = "knn_device_test.pt"
#     print(f"\n3. Saving model (trained on CPU)...")
#     knn1.save(filepath)
    
#     # Load on different device
#     if torch.cuda.is_available():
#         print("\n4. Loading on CUDA...")
#         knn2 = KNNClassifier(device=torch.device('cuda'))
#         knn2.load(filepath)
#         print(f"   Device: {knn2.device}")
#         print(f"   X_train device: {knn2.X_train.device}")
        
#         score2 = knn2.score(X_test, y_test)
#         print(f"   Accuracy: {score2:.4f}")
#         print(f"   Scores match: {score1 == score2}")
#     else:
#         print("\n4. Loading on CPU (CUDA not available)...")
#         knn2 = KNNClassifier(device=torch.device('cpu'))
#         knn2.load(filepath)
#         print(f"   Device: {knn2.device}")
        
#         score2 = knn2.score(X_test, y_test)
#         print(f"   Accuracy: {score2:.4f}")
#         print(f"   Scores match: {score1 == score2}")
    
#     # Cleanup
#     if os.path.exists(filepath):
#         os.remove(filepath)
#         print(f"\n5. Cleaned up: {filepath}")
    
#     print("\n✓ TEST 4 PASSED\n")


# def test_full_workflow():
#     """Test complete workflow: train -> save -> load -> device move -> predict."""
#     print("=" * 60)
#     print("TEST 5: Full Workflow Integration")
#     print("=" * 60)
    
    
    
#     # Create data
#     print("\n1. Creating dataset...")
#     X_train, X_test, y_train, y_test = create_classification_data()
#     print(f"   Training samples: {len(X_train)}")
#     print(f"   Test samples: {len(X_test)}")
#     print(f"   Features: {X_train.shape[1]}")
#     print(f"   Classes: {np.unique(y_train)}")
    
#     # Train model
#     print("\n2. Training KNN Classifier...")
#     knn = KNNClassifier(
#         n_neighbors=7,
#         weights='distance',
#         metric='euclidean'
#     )
#     knn.fit(X_train, y_train, verbose=True)
    
#     # Initial predictions
#     print("\n3. Making initial predictions...")
#     pred_initial = knn.predict(X_test)
#     score_initial = knn.score(X_test, y_test)
#     print(f"   Accuracy: {score_initial:.4f}")
#     print(f"   Sample predictions: {pred_initial[:10]}")
    
#     # Get probabilities
#     proba = knn.predict_proba(X_test[:5])
#     print(f"\n   Class probabilities (first 5 samples):")
#     for i, p in enumerate(proba):
#         print(f"   Sample {i}: {p}")
    
#     # Save model
#     filepath = "knn_workflow_test.pt"
#     print(f"\n4. Saving model to {filepath}...")
#     knn.save(filepath)
    
#     # Load and verify
#     print("\n5. Loading model...")
#     knn_loaded = KNNClassifier()
#     knn_loaded.load(filepath)
    
#     pred_loaded = knn_loaded.predict(X_test)
#     score_loaded = knn_loaded.score(X_test, y_test)
#     print(f"   Accuracy: {score_loaded:.4f}")
#     print(f"   Predictions match: {np.array_equal(pred_initial, pred_loaded)}")
    
#     # Move to CPU and verify
#     print("\n6. Moving to CPU and verifying...")
#     knn_loaded.to('cpu')
#     pred_cpu = knn_loaded.predict(X_test)
#     score_cpu = knn_loaded.score(X_test, y_test)
#     print(f"   Accuracy on CPU: {score_cpu:.4f}")
#     print(f"   All predictions consistent: {np.array_equal(pred_initial, pred_cpu)}")
    
#     # Get model info
#     print("\n7. Model information:")
#     params = knn_loaded.get_params()
#     print(f"   Parameters: {params}")

# def main():
#     """Run all tests."""
#     print("\n" + "=" * 60)
#     print("APRIORI SAVE/LOAD/TO METHODS TEST SUITE")
#     print("=" * 60 + "\n")
    
#     try:
#         # Run all tests
#         test_knn_classifier_save_load()
#         test_knn_regressor_save_load()
#         test_device_movement()
#         test_save_load_with_device_change()
#         test_full_workflow()
        
#         # Summary
#         print("=" * 60)
#         print("ALL TESTS PASSED SUCCESSFULLY! ✓")
#         print("=" * 60)
        
#     except Exception as e:
#         print(f"\n❌ TEST FAILED: {str(e)}")
#         import traceback
#         traceback.print_exc()
        



# if __name__ == '__main__':
#     main()
    
# PS D:\Auto_ML\new_auto_ml> & C:/Users/Smile/anaconda3/envs/tensorflow/python.exe d:/Auto_ML/new_auto_ml/lightning_ml/neighbours.py

# ============================================================
# APRIORI SAVE/LOAD/TO METHODS TEST SUITE
# ============================================================

# ============================================================
# TEST 1: KNNClassifier Save and Load
# ============================================================

# 1. Creating classification dataset...
#    Train: (140, 10), Test: (60, 10)
#    Classes: [0 1 2]

# 2. Training original KNN Classifier...
# KNN Classifier fitted: 140 samples, 3 classes
#    Original model accuracy: 0.7500
#    Sample predictions: [2 2 2 1 1]

# 3. Saving model to knn_classifier_test.pt...
#    ✓ Model saved successfully

# 4. Loading model into new instance...
#    ✓ Model loaded successfully
#    Loaded model accuracy: 0.7500
#    Sample predictions: [2 2 2 1 1]

# 5. Comparing models:
#    Scores match: True
#    Predictions match: True
#    n_neighbors match: True
#    Training data shape match: True

# 6. Cleaned up: knn_classifier_test.pt

# ✓ TEST 1 PASSED

# ============================================================
# TEST 2: KNNRegressor Save and Load
# ============================================================

# 1. Creating regression dataset...
#    Train: (140, 10), Test: (60, 10)

# 2. Training original KNN Regressor...
# KNN fitted with 140 training samples
#    Original model R² score: 0.6698
#    Sample predictions: [ 95.00651  -91.81892   27.182747  13.634294 -90.2181  ]

# 3. Saving model to knn_regressor_test.pt...
#    ✓ Model saved successfully

# 4. Loading model into new instance...
#    ✓ Model loaded successfully
#    Loaded model R² score: 0.6698
#    Sample predictions: [ 95.00651  -91.81892   27.182747  13.634294 -90.2181  ]

# 5. Comparing models:
#    Scores match: True
#    Predictions match: True
#    Parameters match: True

# 6. Cleaned up: knn_regressor_test.pt

# ✓ TEST 2 PASSED

# ============================================================
# TEST 3: Device Movement (to method)
# ============================================================

# 1. Creating dataset...

# 2. Creating and fitting KNN Classifier...
#    Initial device: cuda
#    Initial accuracy: 0.7500

# 3. Moving to CPU...
#    Current device: cpu
#    X_train device: cpu
#    y_train device: cpu
#    Accuracy after move: 0.7500
#    Scores match: True

# 4. Moving to CUDA...
#    Current device: cuda
#    X_train device: cuda:0
#    Accuracy on CUDA: 0.7500
#    Scores match: True
#    Moved back to: cpu

# ✓ TEST 3 PASSED

# ============================================================
# TEST 4: Save/Load with Device Change
# ============================================================

# 1. Creating dataset...

# 2. Training on CPU...
#    Device: cpu
#    Accuracy: 0.7500

# 3. Saving model (trained on CPU)...

# 4. Loading on CUDA...
#    Device: cuda
#    X_train device: cuda:0
#    Accuracy: 0.7500
#    Scores match: True


# 2. Training KNN Classifier...
# KNN Classifier fitted: 140 samples, 3 classes

# 3. Making initial predictions...
#    Accuracy: 0.6833
#    Sample predictions: [2 2 2 1 1 0 0 0 2 1]

#    Class probabilities (first 5 samples):
#    Sample 0: [0.        0.4056341 0.5943659]
#    Sample 1: [0. 0. 1.]
#    Sample 2: [0.16091835 0.         0.83908165]
#    Sample 3: [0.24692689 0.61626345 0.13680968]
#    Sample 4: [0. 1. 0.]

# 4. Saving model to knn_workflow_test.pt...

# 5. Loading model...
#    Accuracy: 0.6833
#    Predictions match: True

# 6. Moving to CPU and verifying...
#    Accuracy on CPU: 0.6833
#    All predictions consistent: True

# 7. Model information:
#    Parameters: {'device': 'cpu', 'is_fitted': True, 'n_neighbors': 7, 'weights': 'distance', 'metric': 'euclidean'}
# ============================================================
# ALL TESTS PASSED SUCCESSFULLY! ✓
# ============================================================
# PS D:\Auto_ML\new_auto_ml> 






