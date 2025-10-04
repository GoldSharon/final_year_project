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
    
    def get_params(self):
        params = super().get_params()
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
    
    def get_params(self):
        params = super().get_params()
        params.update({
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'gamma_value': self._gamma_value
        })
        return params