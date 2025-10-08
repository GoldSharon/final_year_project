"""
Support Vector Machine (SVM) Implementation using PyTorch
Neural network approximation of SVM with multiple kernel support
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Literal
from .base_model import BaseNeuralModel


# ============================================================================
# KERNEL FUNCTIONS
# ============================================================================

class KernelLayer(nn.Module):
    """Base kernel layer for SVM."""
    
    def __init__(self, kernel: str, gamma: float = 1.0, degree: int = 3, 
                 coef0: float = 0.0):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
    
    def linear_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Linear kernel: K(x, y) = x^T y"""
        return torch.matmul(x, y.T)
    
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)"""
        # Compute pairwise squared distances
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.T)
        return torch.exp(-self.gamma * dist)
    
    def polynomial_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree"""
        return (self.gamma * torch.matmul(x, y.T) + self.coef0) ** self.degree
    
    def sigmoid_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sigmoid kernel: K(x, y) = tanh(gamma * x^T y + coef0)"""
        return torch.tanh(self.gamma * torch.matmul(x, y.T) + self.coef0)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply kernel function."""
        if self.kernel == 'linear':
            return self.linear_kernel(x, y)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(x, y)
        elif self.kernel == 'poly':
            return self.polynomial_kernel(x, y)
        elif self.kernel == 'sigmoid':
            return self.sigmoid_kernel(x, y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")


# ============================================================================
# SVM NEURAL NETWORKS
# ============================================================================

class SVMNet(nn.Module):
    """Neural network for SVM approximation with kernel support."""
    
    def __init__(self, n_features: int, n_outputs: int = 1, kernel: str = 'rbf',
                 gamma: float = 1.0, degree: int = 3, coef0: float = 0.0):
        super().__init__()
        self.kernel = kernel
        self.n_features = n_features
        self.n_outputs = n_outputs
        
        if kernel == 'linear':
            # Linear kernel: simple linear layer
            self.net = nn.Linear(n_features, n_outputs)
            self.use_kernel_layer = False
        else:
            # Nonlinear kernels: use kernel-based approximation with MLP
            self.use_kernel_layer = True
            self.kernel_layer = KernelLayer(kernel, gamma, degree, coef0)
            
            # Store support vectors (will be set during training)
            self.register_buffer('support_vectors', None)
            
            # MLP for kernel approximation
            hidden_size = max(64, n_features * 2)
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, n_outputs)
            )
    
    def forward(self, x):
        return self.net(x)
    
    def set_support_vectors(self, sv: torch.Tensor):
        """Set support vectors for kernel-based prediction."""
        self.support_vectors = sv


# ============================================================================
# SVM REGRESSOR
# ============================================================================

class SVMRegressor(BaseNeuralModel):
    """
    Support Vector Machine Regressor using PyTorch.
    
    Supports multiple kernel functions:
    - 'linear': Linear kernel K(x, y) = x^T y
    - 'rbf': RBF kernel K(x, y) = exp(-gamma * ||x - y||^2)
    - 'poly': Polynomial kernel K(x, y) = (gamma * x^T y + coef0)^degree
    - 'sigmoid': Sigmoid kernel K(x, y) = tanh(gamma * x^T y + coef0)
    
    Args:
        C: Regularization parameter (larger C = less regularization)
        epsilon: Epsilon in epsilon-insensitive loss
        kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
        gamma: Kernel coefficient (for rbf/poly/sigmoid). 'scale' or 'auto' or float
        degree: Degree for polynomial kernel
        coef0: Independent term for poly/sigmoid kernels
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        optimizer: Optimizer type
        device: Computation device
    """
    
    def __init__(self,
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 kernel: str = 'rbf',
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 epochs: int = 200,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 optimizer: str = 'adam',
                 device: Optional[torch.device] = None):
        super().__init__(epochs, lr, batch_size, optimizer, device)
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.support_vectors_ = None
        self._gamma_value = None
    
    def _compute_gamma(self, X: torch.Tensor) -> float:
        """Compute gamma value based on input."""
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                variance = X.var().item()
                return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
            elif self.gamma == 'auto':
                return 1.0 / X.shape[1]
            else:
                raise ValueError(f"Unknown gamma value: {self.gamma}")
        else:
            return float(self.gamma)
    
    def _epsilon_insensitive_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Epsilon-insensitive loss for SVR.
        L = max(0, |y_pred - y_true| - epsilon)
        """
        diff = torch.abs(y_pred - y_true)
        loss = torch.clamp(diff - self.epsilon, min=0.0)
        return loss.mean()
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'SVMRegressor':
        """
        Fit SVM regressor.
        
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
        
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        # Compute gamma value
        self._gamma_value = self._compute_gamma(X)
        
        # Initialize model with kernel parameters
        n_features = X.shape[1]
        self.model = SVMNet(
            n_features, 
            n_outputs=1, 
            kernel=self.kernel,
            gamma=self._gamma_value,
            degree=self.degree,
            coef0=self.coef0
        ).to(self.device)
        
        # Setup optimizer
        self._create_optimizer()
        
        # Custom loss function
        self.criterion = self._epsilon_insensitive_loss
        
        # Create data loader
        data_loader = self._create_data_loader(X, y, shuffle=True)
        
        # Training loop
        self._training_history = {'loss': [], 'epoch': []}
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            
            for X_batch, y_batch in data_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Epsilon-insensitive loss
                loss = self.criterion(outputs, y_batch)
                
                # Add L2 regularization (SVM penalty)
                l2_reg = torch.tensor(0., device=self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2)
                
                total_loss_with_reg = loss + (1.0 / (2.0 * self.C)) * l2_reg
                
                # Backward pass
                total_loss_with_reg.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            self._training_history['loss'].append(avg_loss)
            self._training_history['epoch'].append(epoch)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Store support vectors (training data)
        self.support_vectors_ = X.detach()
        if hasattr(self.model, 'set_support_vectors'):
            self.model.set_support_vectors(self.support_vectors_)
        
        self._update_fit_info(X)
        
        if verbose:
            print(f"SVM Regressor fitted with {self.kernel} kernel")
            print(f"Gamma value: {self._gamma_value:.6f}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        with torch.no_grad():
            X = self._to_tensor(X, dtype=torch.float32)
            predictions = self.model(X)
            return self._to_numpy(predictions).flatten()
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate R² score."""
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        y = y.flatten()
        
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def get_params(self, deep: bool = True):
        """Get model parameters including SVM-specific attributes."""
        params = super().get_params(deep=deep)
        params.update({
            'C': self.C,
            'epsilon': self.epsilon,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'gamma_value': self._gamma_value
        })
        return params
    
    def save_model(self, filepath: str):
        """
        Save SVM regressor model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            import warnings
            warnings.warn("Saving unfitted model")
        
        # Get model state and remove support_vectors from it if present
        model_state = None
        if self.model is not None:
            model_state = self.model.state_dict()
            # Remove support_vectors from model state as we'll save it separately
            if 'support_vectors' in model_state:
                del model_state['support_vectors']
        
        state = {
            'model_state': model_state,
            'params': self.get_params(),
            'training_history': self._training_history,
            'is_fitted': self.is_fitted,
            'model_class': self.__class__.__name__,
            # SVM-specific attributes
            'C': self.C,
            'epsilon': self.epsilon,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            '_gamma_value': self._gamma_value,
            'support_vectors': self.support_vectors_,  # Save separately
            'n_features_in': self.n_features_in_,
            'n_samples_seen': self.n_samples_seen_,
            'epochs': self.epochs,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'optimizer_name': self.optimizer_name
        }
        
        torch.save(state, filepath)
        print(f"SVM Regressor saved to {filepath}")


    def load_model(self, filepath: str):
        """
        Load SVM regressor model from file.
        
        Args:
            filepath: Path to model file
        """
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore SVM-specific attributes
        self.C = state.get('C', 1.0)
        self.epsilon = state.get('epsilon', 0.1)
        self.kernel = state.get('kernel', 'rbf')
        self.gamma = state.get('gamma', 'scale')
        self.degree = state.get('degree', 3)
        self.coef0 = state.get('coef0', 0.0)
        self._gamma_value = state.get('_gamma_value', None)
        self.support_vectors_ = state.get('support_vectors', None)
        self.n_features_in_ = state.get('n_features_in', None)
        self.n_samples_seen_ = state.get('n_samples_seen', 0)
        self.epochs = state.get('epochs', 200)
        self.lr = state.get('lr', 0.001)
        self.batch_size = state.get('batch_size', 32)
        self.optimizer_name = state.get('optimizer_name', 'adam')
        
        # Restore training history
        self.is_fitted = state.get('is_fitted', False)
        self._training_history = state.get('training_history', {'loss': [], 'epoch': []})
        
        # Recreate model if fitted
        if self.is_fitted and self.n_features_in_ is not None:
            self.model = SVMNet(
                self.n_features_in_,
                n_outputs=1,
                kernel=self.kernel,
                gamma=self._gamma_value,
                degree=self.degree,
                coef0=self.coef0
            ).to(self.device)
            
            if state['model_state'] is not None:
                # Load state dict without support_vectors
                self.model.load_state_dict(state['model_state'], strict=False)
            
            # Restore support vectors after loading the model
            if self.support_vectors_ is not None:
                self.support_vectors_ = self.support_vectors_.to(self.device)
                if hasattr(self.model, 'set_support_vectors'):
                    self.model.set_support_vectors(self.support_vectors_)
        
        print(f"SVM Regressor loaded from {filepath}")
    
    def to(self, device: Union[str, torch.device]):
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cuda', 'cpu', 'mps')
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        
        if self.model is not None:
            self.model = self.model.to(self.device)
        
        if self.support_vectors_ is not None:
            self.support_vectors_ = self.support_vectors_.to(self.device)
        
        print(f"SVM Regressor moved to device: {self.device}")
        return self


# ============================================================================
# SVM CLASSIFIER
# ============================================================================

class SVMClassifier(BaseNeuralModel):
    """
    Support Vector Machine Classifier using PyTorch.
    
    Supports multiple kernel functions:
    - 'linear': Linear kernel K(x, y) = x^T y
    - 'rbf': RBF kernel K(x, y) = exp(-gamma * ||x - y||^2)
    - 'poly': Polynomial kernel K(x, y) = (gamma * x^T y + coef0)^degree
    - 'sigmoid': Sigmoid kernel K(x, y) = tanh(gamma * x^T y + coef0)
    
    Args:
        C: Regularization parameter (larger C = less regularization)
        kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
        gamma: Kernel coefficient. 'scale' or 'auto' or float
        degree: Degree for polynomial kernel
        coef0: Independent term for poly/sigmoid kernels
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        optimizer: Optimizer type
        device: Computation device
    """
    
    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 epochs: int = 200,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 optimizer: str = 'adam',
                 device: Optional[torch.device] = None):
        super().__init__(epochs, lr, batch_size, optimizer, device)
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.classes_ = None
        self.n_classes_ = None
        self.support_vectors_ = None
        self._gamma_value = None
    
    def _compute_gamma(self, X: torch.Tensor) -> float:
        """Compute gamma value based on input."""
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                variance = X.var().item()
                return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
            elif self.gamma == 'auto':
                return 1.0 / X.shape[1]
            else:
                raise ValueError(f"Unknown gamma value: {self.gamma}")
        else:
            return float(self.gamma)
    
    def _multiclass_hinge_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-class hinge loss."""
        batch_size = y_pred.shape[0]
        correct_class_scores = y_pred[torch.arange(batch_size), y_true].view(-1, 1)
        
        margins = y_pred - correct_class_scores + 1.0
        margins[torch.arange(batch_size), y_true] = 0.0
        
        loss = torch.clamp(margins, min=0.0)
        return loss.sum() / batch_size
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'SVMClassifier':
        """
        Fit SVM classifier.
        
        Args:
            X: Training features
            y: Training labels
            verbose: Print progress
            
        Returns:
            self: Fitted model
        """
        X = self._to_tensor(X, dtype=torch.float32)
        y_np = y if isinstance(y, np.ndarray) else self._to_numpy(y)
        
        # Get unique classes
        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        
        y = self._to_tensor(y_np, dtype=torch.long)
        
        self._validate_input(X, y)
        
        # Compute gamma value
        self._gamma_value = self._compute_gamma(X)
        
        # Initialize model with kernel parameters
        n_features = X.shape[1]
        self.model = SVMNet(
            n_features, 
            n_outputs=self.n_classes_, 
            kernel=self.kernel,
            gamma=self._gamma_value,
            degree=self.degree,
            coef0=self.coef0
        ).to(self.device)
        
        # Setup optimizer
        self._create_optimizer()
        
        # Use hinge loss for SVM
        self.criterion = self._multiclass_hinge_loss
        
        # Create data loader
        data_loader = self._create_data_loader(X, y, shuffle=True)
        
        # Training loop
        self._training_history = {'loss': [], 'epoch': []}
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            
            for X_batch, y_batch in data_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Hinge loss
                loss = self.criterion(outputs, y_batch)
                
                # Add L2 regularization
                l2_reg = torch.tensor(0., device=self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2)
                
                total_loss_with_reg = loss + (1.0 / (2.0 * self.C)) * l2_reg
                
                # Backward pass
                total_loss_with_reg.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            self._training_history['loss'].append(avg_loss)
            self._training_history['epoch'].append(epoch)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Store support vectors (training data)
        self.support_vectors_ = X.detach()
        if hasattr(self.model, 'set_support_vectors'):
            self.model.set_support_vectors(self.support_vectors_)
        
        self._update_fit_info(X)
        
        if verbose:
            print(f"SVM Classifier fitted with {self.n_classes_} classes")
            print(f"Kernel: {self.kernel}, Gamma: {self._gamma_value:.6f}")
        
        return self
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class probabilities using softmax."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        with torch.no_grad():
            X = self._to_tensor(X, dtype=torch.float32)
            outputs = self.model(X)
            probas = torch.softmax(outputs, dim=1)
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
    
    def get_params(self, deep: bool = True):
        """Get model parameters including SVM-specific attributes."""
        params = super().get_params(deep=deep)
        params.update({
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'gamma_value': self._gamma_value,
            'n_classes': self.n_classes_
        })
        return params
    
    def save_model(self, filepath: str):
        """
        Save SVM classifier model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            import warnings
            warnings.warn("Saving unfitted model")
        
        # Get model state and remove support_vectors from it if present
        model_state = None
        if self.model is not None:
            model_state = self.model.state_dict()
            # Remove support_vectors from model state as we'll save it separately
            if 'support_vectors' in model_state:
                del model_state['support_vectors']
        
        state = {
            'model_state': model_state,
            'params': self.get_params(),
            'training_history': self._training_history,
            'is_fitted': self.is_fitted,
            'model_class': self.__class__.__name__,
            # SVM-specific attributes
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            '_gamma_value': self._gamma_value,
            'support_vectors': self.support_vectors_,  # Save separately
            'classes': self.classes_,
            'n_classes': self.n_classes_,
            'n_features_in': self.n_features_in_,
            'n_samples_seen': self.n_samples_seen_,
            'epochs': self.epochs,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'optimizer_name': self.optimizer_name
        }
        
        torch.save(state, filepath)
        print(f"SVM Classifier saved to {filepath}")


    def load_model(self, filepath: str):
        """
        Load SVM classifier model from file.
        
        Args:
            filepath: Path to model file
        """
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore SVM-specific attributes
        self.C = state.get('C', 1.0)
        self.kernel = state.get('kernel', 'rbf')
        self.gamma = state.get('gamma', 'scale')
        self.degree = state.get('degree', 3)
        self.coef0 = state.get('coef0', 0.0)
        self._gamma_value = state.get('_gamma_value', None)
        self.support_vectors_ = state.get('support_vectors', None)
        self.classes_ = state.get('classes', None)
        self.n_classes_ = state.get('n_classes', None)
        self.n_features_in_ = state.get('n_features_in', None)
        self.n_samples_seen_ = state.get('n_samples_seen', 0)
        self.epochs = state.get('epochs', 200)
        self.lr = state.get('lr', 0.001)
        self.batch_size = state.get('batch_size', 32)
        self.optimizer_name = state.get('optimizer_name', 'adam')
        
        # Restore training history
        self.is_fitted = state.get('is_fitted', False)
        self._training_history = state.get('training_history', {'loss': [], 'epoch': []})
        
        # Recreate model if fitted
        if self.is_fitted and self.n_features_in_ is not None and self.n_classes_ is not None:
            self.model = SVMNet(
                self.n_features_in_,
                n_outputs=self.n_classes_,
                kernel=self.kernel,
                gamma=self._gamma_value,
                degree=self.degree,
                coef0=self.coef0
            ).to(self.device)
            
            if state['model_state'] is not None:
                # Load state dict without support_vectors (using strict=False)
                self.model.load_state_dict(state['model_state'], strict=False)
            
            # Restore support vectors after loading the model
            if self.support_vectors_ is not None:
                self.support_vectors_ = self.support_vectors_.to(self.device)
                if hasattr(self.model, 'set_support_vectors'):
                    self.model.set_support_vectors(self.support_vectors_)
        
        print(f"SVM Classifier loaded from {filepath}")
    
    def to(self, device: Union[str, torch.device]):
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cuda', 'cpu', 'mps')
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        
        if self.model is not None:
            self.model = self.model.to(self.device)
        
        if self.support_vectors_ is not None:
            self.support_vectors_ = self.support_vectors_.to(self.device)
        
        print(f"SVM Classifier moved to device: {self.device}")
        return self
    
# """
# Test Script for SVM Save/Load/To Methods
# Tests SVMRegressor and SVMClassifier functionality
# """

# import numpy as np
# import torch
# import os
# import tempfile
# from sklearn.datasets import make_regression, make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, accuracy_score

# # Import your SVM models (adjust import path as needed)
# # from your_module import SVMRegressor, SVMClassifier


# def test_svm_regressor():
#     """Test SVMRegressor save/load/to functionality"""
#     print("=" * 70)
#     print("TESTING SVM REGRESSOR")
#     print("=" * 70)
    
#     # Generate regression data
#     X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     print(f"\nDataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
#     print(f"Features: {X_train.shape[1]}")
    
#     # Test different kernels
#     kernels = ['linear', 'rbf', 'poly']
    
#     for kernel in kernels:
#         print(f"\n{'─' * 70}")
#         print(f"Testing Kernel: {kernel.upper()}")
#         print(f"{'─' * 70}")
        
#         # 1. Train original model
#         print("\n1. Training original model...")
#         model = SVMRegressor(
#             C=1.0,
#             epsilon=0.1,
#             kernel=kernel,
#             gamma='scale',
#             degree=3,
#             epochs=100,
#             lr=0.001,
#             batch_size=32
#         )
        
#         model.fit(X_train, y_train, verbose=False)
#         score_original = model.score(X_test, y_test)
#         pred_original = model.predict(X_test)
        
#         print(f"   ✓ Model fitted successfully")
#         print(f"   ✓ R² Score: {score_original:.4f}")
#         print(f"   ✓ Device: {model.device}")
        
#         # 2. Test save_model
#         print("\n2. Testing save_model()...")
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
#         temp_path = temp_file.name
#         temp_file.close()
        
#         try:
#             model.save_model(temp_path)
#             print(f"   ✓ Model saved to: {temp_path}")
#             print(f"   ✓ File size: {os.path.getsize(temp_path) / 1024:.2f} KB")
            
#             # 3. Test load_model
#             print("\n3. Testing load_model()...")
#             model_loaded = SVMRegressor()
#             model_loaded.load_model(temp_path)
            
#             print(f"   ✓ Model loaded successfully")
#             print(f"   ✓ Kernel: {model_loaded.kernel}")
#             print(f"   ✓ Gamma value: {model_loaded._gamma_value:.6f}")
#             print(f"   ✓ C: {model_loaded.C}")
#             print(f"   ✓ Epsilon: {model_loaded.epsilon}")
            
#             # Verify predictions match
#             pred_loaded = model_loaded.predict(X_test)
#             score_loaded = model_loaded.score(X_test, y_test)
            
#             pred_diff = np.abs(pred_original - pred_loaded).max()
#             score_diff = abs(score_original - score_loaded)
            
#             print(f"\n4. Verification:")
#             print(f"   ✓ Loaded R² Score: {score_loaded:.4f}")
#             print(f"   ✓ Max prediction difference: {pred_diff:.10f}")
#             print(f"   ✓ Score difference: {score_diff:.10f}")
            
#             if pred_diff < 1e-5 and score_diff < 1e-5:
#                 print(f"   ✅ PASS: Predictions match perfectly!")
#             else:
#                 print(f"   ⚠️  WARNING: Small differences detected")
            
#             # 5. Test to() method - CPU
#             print("\n5. Testing to() method (CPU)...")
#             model_cpu = model_loaded.to('cpu')
#             pred_cpu = model_cpu.predict(X_test[:5])
#             print(f"   ✓ Model moved to CPU")
#             print(f"   ✓ Device: {model_cpu.device}")
#             print(f"   ✓ Predictions on CPU: {pred_cpu[:3]}")
            
#             # 6. Test to() method - CUDA (if available)
#             if torch.cuda.is_available():
#                 print("\n6. Testing to() method (CUDA)...")
#                 model_cuda = model_loaded.to('cuda')
#                 pred_cuda = model_cuda.predict(X_test[:5])
#                 print(f"   ✓ Model moved to CUDA")
#                 print(f"   ✓ Device: {model_cuda.device}")
#                 print(f"   ✓ Predictions on CUDA: {pred_cuda[:3]}")
                
#                 # Move back to CPU
#                 model_cuda.to('cpu')
#                 print(f"   ✓ Model moved back to CPU")
#             else:
#                 print("\n6. CUDA not available, skipping GPU test")
            
#             # 7. Test get_params
#             print("\n7. Testing get_params()...")
#             params = model_loaded.get_params()
#             print(f"   ✓ Retrieved {len(params)} parameters")
#             print(f"   ✓ Key parameters: C={params['C']}, kernel={params['kernel']}")
            
#         finally:
#             # Cleanup
#             if os.path.exists(temp_path):
#                 os.unlink(temp_path)
#             print(f"\n   ✓ Cleanup: Temporary file removed")
    
#     print(f"\n{'═' * 70}")
#     print("✅ SVM REGRESSOR TESTS COMPLETED")
#     print(f"{'═' * 70}\n")


# def test_svm_classifier():
#     """Test SVMClassifier save/load/to functionality"""
#     print("=" * 70)
#     print("TESTING SVM CLASSIFIER")
#     print("=" * 70)
    
#     # Generate classification data
#     X, y = make_classification(n_samples=300, n_features=10, n_informative=8,
#                                n_redundant=2, n_classes=3, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     print(f"\nDataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
#     print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    
#     # Test different kernels
#     kernels = ['linear', 'rbf']
    
#     for kernel in kernels:
#         print(f"\n{'─' * 70}")
#         print(f"Testing Kernel: {kernel.upper()}")
#         print(f"{'─' * 70}")
        
#         # 1. Train original model
#         print("\n1. Training original model...")
#         model = SVMClassifier(
#             C=1.0,
#             kernel=kernel,
#             gamma='scale',
#             epochs=100,
#             lr=0.001,
#             batch_size=32
#         )
        
#         model.fit(X_train, y_train, verbose=False)
#         score_original = model.score(X_test, y_test)
#         pred_original = model.predict(X_test)
#         proba_original = model.predict_proba(X_test)
        
#         print(f"   ✓ Model fitted successfully")
#         print(f"   ✓ Accuracy: {score_original:.4f}")
#         print(f"   ✓ Classes: {model.classes_}")
#         print(f"   ✓ Device: {model.device}")
        
#         # 2. Test save_model
#         print("\n2. Testing save_model()...")
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
#         temp_path = temp_file.name
#         temp_file.close()
        
#         try:
#             model.save_model(temp_path)
#             print(f"   ✓ Model saved to: {temp_path}")
#             print(f"   ✓ File size: {os.path.getsize(temp_path) / 1024:.2f} KB")
            
#             # 3. Test load_model
#             print("\n3. Testing load_model()...")
#             model_loaded = SVMClassifier()
#             model_loaded.load_model(temp_path)
            
#             print(f"   ✓ Model loaded successfully")
#             print(f"   ✓ Kernel: {model_loaded.kernel}")
#             print(f"   ✓ Gamma value: {model_loaded._gamma_value:.6f}")
#             print(f"   ✓ C: {model_loaded.C}")
#             print(f"   ✓ Classes: {model_loaded.classes_}")
#             print(f"   ✓ N Classes: {model_loaded.n_classes_}")
            
#             # Verify predictions match
#             pred_loaded = model_loaded.predict(X_test)
#             proba_loaded = model_loaded.predict_proba(X_test)
#             score_loaded = model_loaded.score(X_test, y_test)
            
#             pred_match = np.array_equal(pred_original, pred_loaded)
#             proba_diff = np.abs(proba_original - proba_loaded).max()
#             score_diff = abs(score_original - score_loaded)
            
#             print(f"\n4. Verification:")
#             print(f"   ✓ Loaded Accuracy: {score_loaded:.4f}")
#             print(f"   ✓ Predictions match: {pred_match}")
#             print(f"   ✓ Max probability difference: {proba_diff:.10f}")
#             print(f"   ✓ Score difference: {score_diff:.10f}")
            
#             if pred_match and proba_diff < 1e-5 and score_diff < 1e-5:
#                 print(f"   ✅ PASS: Predictions match perfectly!")
#             else:
#                 print(f"   ⚠️  WARNING: Small differences detected")
            
#             # 5. Test to() method - CPU
#             print("\n5. Testing to() method (CPU)...")
#             model_cpu = model_loaded.to('cpu')
#             pred_cpu = model_cpu.predict(X_test[:5])
#             proba_cpu = model_cpu.predict_proba(X_test[:5])
#             print(f"   ✓ Model moved to CPU")
#             print(f"   ✓ Device: {model_cpu.device}")
#             print(f"   ✓ Sample predictions: {pred_cpu}")
            
#             # 6. Test to() method - CUDA (if available)
#             if torch.cuda.is_available():
#                 print("\n6. Testing to() method (CUDA)...")
#                 model_cuda = model_loaded.to('cuda')
#                 pred_cuda = model_cuda.predict(X_test[:5])
#                 print(f"   ✓ Model moved to CUDA")
#                 print(f"   ✓ Device: {model_cuda.device}")
#                 print(f"   ✓ Sample predictions: {pred_cuda}")
                
#                 # Move back to CPU
#                 model_cuda.to('cpu')
#                 print(f"   ✓ Model moved back to CPU")
#             else:
#                 print("\n6. CUDA not available, skipping GPU test")
            
#             # 7. Test get_params
#             print("\n7. Testing get_params()...")
#             params = model_loaded.get_params()
#             print(f"   ✓ Retrieved {len(params)} parameters")
#             print(f"   ✓ Key parameters: C={params['C']}, kernel={params['kernel']}")
            
#             # 8. Test training history
#             print("\n8. Testing training history...")
#             history = model_loaded.get_training_history()
#             print(f"   ✓ Training history available: {len(history['loss'])} epochs")
#             print(f"   ✓ Final loss: {history['loss'][-1]:.4f}")
            
#         finally:
#             # Cleanup
#             if os.path.exists(temp_path):
#                 os.unlink(temp_path)
#             print(f"\n   ✓ Cleanup: Temporary file removed")
    
#     print(f"\n{'═' * 70}")
#     print("✅ SVM CLASSIFIER TESTS COMPLETED")
#     print(f"{'═' * 70}\n")


# def test_edge_cases():
#     """Test edge cases and error handling"""
#     print("=" * 70)
#     print("TESTING EDGE CASES")
#     print("=" * 70)
    
#     # 1. Test prediction before fitting
#     print("\n1. Testing prediction before fitting...")
#     try:
#         model = SVMRegressor()
#         X_dummy = np.random.randn(10, 5)
#         model.predict(X_dummy)
#         print("   ❌ FAIL: Should raise error")
#     except ValueError as e:
#         print(f"   ✓ PASS: Raised ValueError as expected")
#         print(f"   ✓ Error message: {str(e)}")
    
#     # 2. Test loading to different device
#     print("\n2. Testing load to different device...")
#     X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    
#     model = SVMRegressor(epochs=20)
#     model.fit(X, y, verbose=False)
    
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
#     temp_path = temp_file.name
#     temp_file.close()
    
#     try:
#         model.save_model(temp_path)
        
#         # Load on CPU explicitly
#         model_cpu = SVMRegressor(device=torch.device('cpu'))
#         model_cpu.load_model(temp_path)
#         pred = model_cpu.predict(X[:5])
        
#         print(f"   ✓ Model loaded on CPU device")
#         print(f"   ✓ Predictions shape: {pred.shape}")
        
#     finally:
#         if os.path.exists(temp_path):
#             os.unlink(temp_path)
    
#     # 3. Test save/load unfitted model
#     print("\n3. Testing save unfitted model...")
#     model = SVMClassifier()
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
#     temp_path = temp_file.name
#     temp_file.close()
    
#     try:
#         import warnings
#         with warnings.catch_warnings(record=True) as w:
#             warnings.simplefilter("always")
#             model.save_model(temp_path)
#             if len(w) > 0:
#                 print(f"   ✓ Warning raised: {w[0].message}")
#             print(f"   ✓ Unfitted model saved successfully")
#     finally:
#         if os.path.exists(temp_path):
#             os.unlink(temp_path)
    
#     print(f"\n{'═' * 70}")
#     print("✅ EDGE CASE TESTS COMPLETED")
#     print(f"{'═' * 70}\n")


# def main():
#     """Run all tests"""
#     print("\n" + "=" * 70)
#     print("SVM SAVE/LOAD/TO COMPREHENSIVE TEST SUITE")
#     print("=" * 70 + "\n")
    
#     try:
#         # Test SVMRegressor
#         test_svm_regressor()
        
#         # Test SVMClassifier
#         test_svm_classifier()
        
#         # Test edge cases
#         test_edge_cases()
        
#         print("\n" + "=" * 70)
#         print("🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
#         print("=" * 70 + "\n")
        
#     except Exception as e:
#         print(f"\n❌ TEST FAILED WITH ERROR:")
#         print(f"   {type(e).__name__}: {str(e)}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()
    
# PS D:\Auto_ML\new_auto_ml> & C:/Users/Smile/anaconda3/envs/tensorflow/python.exe d:/Auto_ML/new_auto_ml/lightning_ml/svm.py

# ======================================================================
# SVM SAVE/LOAD/TO COMPREHENSIVE TEST SUITE
# ======================================================================

# ======================================================================
# TESTING SVM REGRESSOR
# ======================================================================

# Dataset: 140 train samples, 60 test samples
# Features: 10

# ──────────────────────────────────────────────────────────────────────
# Testing Kernel: LINEAR
# ──────────────────────────────────────────────────────────────────────

# 1. Training original model...
#    ✓ Model fitted successfully
#    ✓ R² Score: -0.0034
#    ✓ Device: cuda

# 2. Testing save_model()...
# SVM Regressor saved to C:\Users\Smile\AppData\Local\Temp\tmp8dllz6pw.pth
#    ✓ Model saved to: C:\Users\Smile\AppData\Local\Temp\tmp8dllz6pw.pth
#    ✓ File size: 9.32 KB

# 3. Testing load_model()...
# SVM Regressor loaded from C:\Users\Smile\AppData\Local\Temp\tmp8dllz6pw.pth
#    ✓ Model loaded successfully
#    ✓ Kernel: linear
#    ✓ Gamma value: 0.103763
#    ✓ C: 1.0
#    ✓ Epsilon: 0.1

# 4. Verification:
#    ✓ Loaded R² Score: -0.0034
#    ✓ Max prediction difference: 0.0000000000
#    ✓ Score difference: 0.0000000000
#    ✅ PASS: Predictions match perfectly!

# 5. Testing to() method (CPU)...
# SVM Regressor moved to device: cpu
#    ✓ Model moved to CPU
#    ✓ Device: cpu
#    ✓ Predictions on CPU: [ 1.544448  -1.1973228 -0.0893295]

# 6. Testing to() method (CUDA)...
# SVM Regressor moved to device: cuda
#    ✓ Model moved to CUDA
#    ✓ Device: cuda
#    ✓ Predictions on CUDA: [ 1.5444483 -1.1973228 -0.0893295]
# SVM Regressor moved to device: cpu
#    ✓ Model moved back to CPU

# 7. Testing get_params()...
#    ✓ Retrieved 14 parameters
#    ✓ Key parameters: C=1.0, kernel=linear

#    ✓ Cleanup: Temporary file removed

# ──────────────────────────────────────────────────────────────────────
# Testing Kernel: RBF
# ──────────────────────────────────────────────────────────────────────

# 1. Training original model...
#    ✓ Model fitted successfully
#    ✓ R² Score: -0.0116
#    ✓ Device: cuda

# 2. Testing save_model()...
# SVM Regressor saved to C:\Users\Smile\AppData\Local\Temp\tmprjsqi194.pth
#    ✓ Model saved to: C:\Users\Smile\AppData\Local\Temp\tmprjsqi194.pth
#    ✓ File size: 21.50 KB

# 3. Testing load_model()...
# SVM Regressor loaded from C:\Users\Smile\AppData\Local\Temp\tmprjsqi194.pth
#    ✓ Model loaded successfully
#    ✓ Kernel: rbf
#    ✓ Gamma value: 0.103763
#    ✓ C: 1.0
#    ✓ Epsilon: 0.1

# 4. Verification:
#    ✓ Loaded R² Score: -0.0116
#    ✓ Max prediction difference: 0.0000000000
#    ✓ Score difference: 0.0000000000
#    ✅ PASS: Predictions match perfectly!

# 5. Testing to() method (CPU)...
# SVM Regressor moved to device: cpu
#    ✓ Model moved to CPU
#    ✓ Device: cpu
#    ✓ Predictions on CPU: [0.00075391 0.00075391 0.00075391]

# 6. Testing to() method (CUDA)...
# SVM Regressor moved to device: cuda
#    ✓ Model moved to CUDA
#    ✓ Device: cuda
#    ✓ Predictions on CUDA: [0.00075391 0.00075391 0.00075391]
# SVM Regressor moved to device: cpu
#    ✓ Model moved back to CPU

# 7. Testing get_params()...
#    ✓ Retrieved 14 parameters
#    ✓ Key parameters: C=1.0, kernel=rbf

#    ✓ Cleanup: Temporary file removed

# ──────────────────────────────────────────────────────────────────────
# Testing Kernel: POLY
# ──────────────────────────────────────────────────────────────────────

# 1. Training original model...
#    ✓ Model fitted successfully
#    ✓ R² Score: -0.0116
#    ✓ Device: cuda

# 2. Testing save_model()...
# SVM Regressor saved to C:\Users\Smile\AppData\Local\Temp\tmpco2ujtty.pth
#    ✓ Model saved to: C:\Users\Smile\AppData\Local\Temp\tmpco2ujtty.pth
#    ✓ File size: 21.50 KB

# 3. Testing load_model()...
# SVM Regressor loaded from C:\Users\Smile\AppData\Local\Temp\tmpco2ujtty.pth
#    ✓ Model loaded successfully
#    ✓ Kernel: poly
#    ✓ Gamma value: 0.103763
#    ✓ C: 1.0
#    ✓ Epsilon: 0.1

# 4. Verification:
#    ✓ Loaded R² Score: -0.0116
#    ✓ Max prediction difference: 0.0000000000
#    ✓ Score difference: 0.0000000000
#    ✅ PASS: Predictions match perfectly!

# 5. Testing to() method (CPU)...
# SVM Regressor moved to device: cpu
#    ✓ Model moved to CPU
#    ✓ Device: cpu
#    ✓ Predictions on CPU: [-5.3634572e-05 -5.3634565e-05 -5.3634569e-05]

# 6. Testing to() method (CUDA)...
# SVM Regressor moved to device: cuda
#    ✓ Model moved to CUDA
#    ✓ Device: cuda
#    ✓ Predictions on CUDA: [-5.3634572e-05 -5.3634565e-05 -5.3634569e-05]
# SVM Regressor moved to device: cpu
#    ✓ Model moved back to CPU

# 7. Testing get_params()...
#    ✓ Retrieved 14 parameters
#    ✓ Key parameters: C=1.0, kernel=poly

#    ✓ Cleanup: Temporary file removed

# ══════════════════════════════════════════════════════════════════════
# ✅ SVM REGRESSOR TESTS COMPLETED
# ══════════════════════════════════════════════════════════════════════

# ======================================================================
# TESTING SVM CLASSIFIER
# ======================================================================

# Dataset: 210 train samples, 90 test samples
# Features: 10, Classes: 3

# ──────────────────────────────────────────────────────────────────────
# Testing Kernel: LINEAR
# ──────────────────────────────────────────────────────────────────────

# 1. Training original model...
#    ✓ Model fitted successfully
#    ✓ Accuracy: 0.7222
#    ✓ Classes: [0 1 2]
#    ✓ Device: cuda

# 2. Testing save_model()...
# SVM Classifier saved to C:\Users\Smile\AppData\Local\Temp\tmpy8ebvhhg.pth
#    ✓ Model saved to: C:\Users\Smile\AppData\Local\Temp\tmpy8ebvhhg.pth
#    ✓ File size: 12.32 KB

# 3. Testing load_model()...
# SVM Classifier loaded from C:\Users\Smile\AppData\Local\Temp\tmpy8ebvhhg.pth
#    ✓ Model loaded successfully
#    ✓ Kernel: linear
#    ✓ Gamma value: 0.022704
#    ✓ C: 1.0
#    ✓ Classes: [0 1 2]
#    ✓ N Classes: 3

# 4. Verification:
#    ✓ Loaded Accuracy: 0.7222
#    ✓ Predictions match: True
#    ✓ Max probability difference: 0.0000000000
#    ✓ Score difference: 0.0000000000
#    ✅ PASS: Predictions match perfectly!

# 5. Testing to() method (CPU)...
# SVM Classifier moved to device: cpu
#    ✓ Model moved to CPU
#    ✓ Device: cpu
#    ✓ Sample predictions: [1 1 0 1 0]

# 6. Testing to() method (CUDA)...
# SVM Classifier moved to device: cuda
#    ✓ Model moved to CUDA
#    ✓ Device: cuda
#    ✓ Sample predictions: [1 1 0 1 0]
# SVM Classifier moved to device: cpu
#    ✓ Model moved back to CPU

# 7. Testing get_params()...
#    ✓ Retrieved 14 parameters
#    ✓ Key parameters: C=1.0, kernel=linear

# 8. Testing training history...
#    ✓ Training history available: 100 epochs
#    ✓ Final loss: 0.9594

#    ✓ Cleanup: Temporary file removed

# ──────────────────────────────────────────────────────────────────────
# Testing Kernel: RBF
# ──────────────────────────────────────────────────────────────────────

# 1. Training original model...
#    ✓ Model fitted successfully
#    ✓ Accuracy: 0.7111
#    ✓ Classes: [0 1 2]
#    ✓ Device: cuda

# 2. Testing save_model()...
# SVM Classifier saved to C:\Users\Smile\AppData\Local\Temp\tmp9167s0qg.pth
#    ✓ Model saved to: C:\Users\Smile\AppData\Local\Temp\tmp9167s0qg.pth
#    ✓ File size: 24.75 KB

# 3. Testing load_model()...
# SVM Classifier loaded from C:\Users\Smile\AppData\Local\Temp\tmp9167s0qg.pth
#    ✓ Model loaded successfully
#    ✓ Kernel: rbf
#    ✓ Gamma value: 0.022704
#    ✓ C: 1.0
#    ✓ Classes: [0 1 2]
#    ✓ N Classes: 3

# 4. Verification:
#    ✓ Loaded Accuracy: 0.7111
#    ✓ Predictions match: True
#    ✓ Max probability difference: 0.0000000000
#    ✓ Score difference: 0.0000000000
#    ✅ PASS: Predictions match perfectly!

# 5. Testing to() method (CPU)...
# SVM Classifier moved to device: cpu
#    ✓ Model moved to CPU
#    ✓ Device: cpu
#    ✓ Sample predictions: [0 1 0 1 0]

# 6. Testing to() method (CUDA)...
# SVM Classifier moved to device: cuda
#    ✓ Model moved to CUDA
#    ✓ Device: cuda
#    ✓ Sample predictions: [0 1 0 1 0]
# SVM Classifier moved to device: cpu
#    ✓ Model moved back to CPU

# 7. Testing get_params()...
#    ✓ Retrieved 14 parameters
#    ✓ Key parameters: C=1.0, kernel=rbf

# 8. Testing training history...
#    ✓ Training history available: 100 epochs
#    ✓ Final loss: 1.0172

#    ✓ Cleanup: Temporary file removed

# ══════════════════════════════════════════════════════════════════════
# ✅ SVM CLASSIFIER TESTS COMPLETED
# ══════════════════════════════════════════════════════════════════════

# ======================================================================
# TESTING EDGE CASES
# ======================================================================

# 1. Testing prediction before fitting...
#    ✓ PASS: Raised ValueError as expected
#    ✓ Error message: Model must be fitted before prediction

# 2. Testing load to different device...
# SVM Regressor saved to C:\Users\Smile\AppData\Local\Temp\tmp246wfnws.pth
# SVM Regressor loaded from C:\Users\Smile\AppData\Local\Temp\tmp246wfnws.pth
#    ✓ Model loaded on CPU device
#    ✓ Predictions shape: (5,)

# 3. Testing save unfitted model...
# SVM Classifier saved to C:\Users\Smile\AppData\Local\Temp\tmp4tpxs7sm.pth
#    ✓ Warning raised: Saving unfitted model
#    ✓ Unfitted model saved successfully

# ══════════════════════════════════════════════════════════════════════
# ✅ EDGE CASE TESTS COMPLETED
# ══════════════════════════════════════════════════════════════════════


# ======================================================================
# 🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉
# ======================================================================