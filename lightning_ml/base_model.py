"""
Base Model for PyTorch ML Module
Provides common interface for all ML models (supervised and unsupervised)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple
from abc import ABC, abstractmethod
import warnings


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    Provides common interface for:
    - Model training (fit)
    - Prediction (predict)
    - Model persistence (save/load)
    - Device management (GPU/CPU)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize base model.
        
        Args:
            device: torch.device for computation (None = auto-detect)
        """
        self.device = device if device is not None else self._get_default_device()
        self.model = None
        self.is_fitted = False
        self._training_history = {
            'loss': [],
            'epoch': []
        }
        
    def _get_default_device(self) -> torch.device:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')  # Apple Silicon
        else:
            return torch.device('cpu')
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor], 
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Convert input data to PyTorch tensor.
        
        Args:
            data: Input data (numpy array or tensor)
            dtype: Target data type
            
        Returns:
            PyTorch tensor on the correct device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=dtype)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device, dtype=dtype)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Numpy array
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _validate_input(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Validate input data shapes and types.
        
        Args:
            X: Feature tensor
            y: Target tensor (optional)
        """
        if X.dim() == 1:
            raise ValueError("X must be 2D array (n_samples, n_features)")
        
        if y is not None:
            if len(X) != len(y):
                raise ValueError(f"X and y must have same number of samples. "
                               f"Got X: {len(X)}, y: {len(y)}")
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor], 
            **kwargs) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional parameters
            
        Returns:
            self: Fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions as numpy array
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'device': str(self.device),
            'is_fitted': self.is_fitted
        }
    
    def set_params(self, **params):
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def save_model(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            warnings.warn("Saving unfitted model")
        
        state = {
            'model_state': self.model.state_dict() if self.model is not None else None,
            'params': self.get_params(),
            'training_history': self._training_history,
            'is_fitted': self.is_fitted
        }
        
        torch.save(state, filepath)
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        state = torch.load(filepath, map_location=self.device)
        
        if self.model is not None and state['model_state'] is not None:
            self.model.load_state_dict(state['model_state'])
        
        self.is_fitted = state.get('is_fitted', False)
        self._training_history = state.get('training_history', {'loss': [], 'epoch': []})
        
        # Restore parameters
        params = state.get('params', {})
        for key, value in params.items():
            if hasattr(self, key) and key != 'is_fitted':
                setattr(self, key, value)
    
    def get_training_history(self) -> Dict[str, list]:
        """
        Get training history.
        
        Returns:
            Dictionary with loss and epoch history
        """
        return self._training_history.copy()
    
    def to(self, device: Union[str, torch.device]):
        """
        Move model to specified device.
        
        Args:
            device: Target device
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self


class BaseSupervisedModel(BaseModel):
    """
    Base class for supervised learning models (classification and regression).
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
    
    def _update_fit_info(self, X: torch.Tensor):
        """Update metadata after fitting."""
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.is_fitted = True
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute model score.
        
        Args:
            X: Test features
            y: True labels/values
            
        Returns:
            Model score (accuracy for classification, R² for regression)
        """
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        # This should be overridden in child classes
        return 0.0


class BaseUnsupervisedModel(BaseModel):
    """
    Base class for unsupervised learning models (clustering, association).
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y=None, **kwargs) -> 'BaseUnsupervisedModel':
        """
        Fit unsupervised model (y is ignored).
        
        Args:
            X: Training features
            y: Ignored (for API consistency)
            **kwargs: Additional parameters
            
        Returns:
            self: Fitted model
        """
        # This will be implemented by child classes
        pass
    
    def fit_predict(self, X: Union[np.ndarray, torch.Tensor], 
                    y=None) -> np.ndarray:
        """
        Fit model and return predictions in one step.
        
        Args:
            X: Training features
            y: Ignored
            
        Returns:
            Cluster labels or predictions
        """
        self.fit(X, y)
        return self.predict(X)


class BaseNeuralModel(BaseSupervisedModel):
    """
    Base class for neural network-based models.
    Handles common training loop, optimization, and batching.
    """
    
    def __init__(self, 
                 epochs: int = 100,
                 lr: float = 0.01,
                 batch_size: int = 32,
                 optimizer: str = 'adam',
                 device: Optional[torch.device] = None):
        """
        Initialize neural model.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            device: Computation device
        """
        super().__init__(device)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer_name = optimizer
        self.optimizer = None
        self.criterion = None
    
    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        if self.model is None:
            raise ValueError("Model must be initialized before creating optimizer")
        
        if self.optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _create_data_loader(self, X: torch.Tensor, y: torch.Tensor, 
                           shuffle: bool = True):
        """Create PyTorch DataLoader."""
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle
        )
    
    def _train_epoch(self, data_loader) -> float:
        """
        Train for one epoch.
        
        Args:
            data_loader: PyTorch DataLoader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in data_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'epochs': self.epochs,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer_name
        })
        return params


class BaseTreeModel(BaseSupervisedModel):
    """
    Base class for tree-based models (Decision Trees, Random Forests).
    Note: Tree models typically don't use PyTorch gradients but we keep
    the interface consistent.
    """
    
    def __init__(self, 
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 device: Optional[torch.device] = None):
        """
        Initialize tree model.
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf node
            device: Computation device
        """
        super().__init__(device)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.feature_importances_ = None
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        })
        return params