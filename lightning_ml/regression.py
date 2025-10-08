"""
Linear, Logistic, Regression Models using PyTorch
Full implementation with GPU support and proper training loops
Includes save/load/to methods for model persistence and device management
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional
from .base_model import BaseNeuralModel
from .base_model import BaseSupervisedModel


# ============================================================================
# LINEAR REGRESSION
# ============================================================================

class LinearRegressionNet(nn.Module):
    """Neural network for linear regression."""
    
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)

"""
Ordinary Least Squares (OLS) Linear Regression
Compatible with BaseSupervisedModel
"""

class LinearRegression(BaseSupervisedModel):
    """
    Linear Regression using Ordinary Least Squares (OLS).
    
    Formula:
        w = (XᵀX)^(-1) Xᵀ y
        y_pred = Xw + b

    Args:
        fit_intercept: Whether to include intercept term (default: True)
        device: Computation device
    """
    
    def __init__(self, fit_intercept: bool = True, device: Optional[torch.device] = None):
        super().__init__(device)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self._XTX_inv = None  # cached for analysis (optional)
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor], 
            verbose: bool = False) -> 'LinearRegression':
        """
        Fit OLS linear regression model analytically.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,) or (n_samples, 1)
            verbose: Print fitting info

        Returns:
            self: Fitted model
        """
        # Convert inputs
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y, dtype=torch.float32)
        
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze()
            if y.dim() != 1:
                raise ValueError("y must be 1D or a column vector")
        
        # Validate input
        self._validate_input(X, y)
        
        # Store original dimensions BEFORE modifying X
        n_samples, n_features = X.shape
        X_augmented = X  # Keep reference to augmented version

        # Add intercept term if needed
        if self.fit_intercept:
            ones = torch.ones((X.shape[0], 1), device=self.device)
            X_augmented = torch.cat([ones, X], dim=1)

        # Compute closed-form solution: (XᵀX)^(-1) Xᵀ y
        XT = X_augmented.T
        XTX = XT @ X_augmented
        try:
            XTX_inv = torch.linalg.inv(XTX)
        except RuntimeError:
            # Use pseudo-inverse for singular matrices
            XTX_inv = torch.linalg.pinv(XTX)
        
        w = XTX_inv @ XT @ y

        # Store coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = w[0].item()
            self.coef_ = self._to_numpy(w[1:])
        else:
            self.intercept_ = 0.0
            self.coef_ = self._to_numpy(w)

        # Cache inverse for statistical analysis
        self._XTX_inv = self._to_numpy(XTX_inv)
        
        # Update fit metadata with ORIGINAL feature dimensions
        self.n_features_in_ = n_features
        self.n_samples_seen_ = n_samples
        self.is_fitted = True
        
        if verbose:
            print(f"LinearRegression fitted with {n_features} features on {n_samples} samples")
            print(f"Intercept: {self.intercept_:.4f}")
            print(f"Coefficients shape: {self.coef_.shape}")
        
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict target values.

        Args:
            X: Input features (n_samples, n_features)
        Returns:
            Predicted values as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        X = self._to_tensor(X, dtype=torch.float32)
        
        # Validate feature count
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with "
                f"{self.n_features_in_} features"
            )
        
        # Convert coefficients to tensor
        coef_tensor = self._to_tensor(self.coef_, dtype=torch.float32)
        
        # Compute predictions: y = Xw + b
        y_pred = X @ coef_tensor + self.intercept_
        
        return self._to_numpy(y_pred).flatten()

    def score(self, X: Union[np.ndarray, torch.Tensor],
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute R² (coefficient of determination) score.

        Args:
            X: Test features
            y: True targets
        Returns:
            R² score (1.0 is perfect prediction, 0.0 is baseline)
        """
        y_pred = self.predict(X)

        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        y = y.flatten()

        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        
        if ss_tot == 0:
            # Perfect prediction or constant target
            return 1.0 if ss_res == 0 else 0.0
        
        return 1.0 - (ss_res / ss_tot)
    
    def get_params(self, deep: bool = True):
        """Return model parameters."""
        params = super().get_params(deep=deep)
        params.update({
            'fit_intercept': self.fit_intercept,
            'coef_': self.coef_,
            'intercept_': self.intercept_
        })
        return params
    
    def save(self, filepath: str):
        """Save model to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'alpha': self.alpha,
            'n_features_in_': self.n_features_in_,
            'n_samples_seen_': self.n_samples_seen_,
            'is_fitted': self.is_fitted,
            'epochs': self.epochs,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'optimizer_name': self.optimizer_name,
            'training_history': self._training_history,
            'device': str(self.device),
            'model_class': self.__class__.__name__
        }
        
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.alpha = state['alpha']
        self.epochs = state['epochs']
        self.lr = state['lr']
        self.batch_size = state['batch_size']
        self.optimizer_name = state['optimizer_name']
        self.n_features_in_ = state['n_features_in_']
        self.n_samples_seen_ = state['n_samples_seen_']
        self.is_fitted = state['is_fitted']
        self.coef_ = state['coef_']
        self.intercept_ = state['intercept_']
        self._training_history = state['training_history']
        
        # Recreate model
        self.model = LinearRegressionNet(self.n_features_in_).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
    
    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self

    def get_params(self, deep: bool = True):
        params = super().get_params(deep=deep)
        params.update({
            'alpha': self.alpha,
            'coef_': self.coef_,
            'intercept_': self.intercept_
        })
        return params
    
    def get_residuals(self, X: Union[np.ndarray, torch.Tensor],
                      y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute residuals (errors) for given data.
        
        Args:
            X: Features
            y: True targets
        
        Returns:
            Residuals: y - y_pred
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing residuals.")
        
        y_pred = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        return y.flatten() - y_pred
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        state = {
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'fit_intercept': self.fit_intercept,
            'n_features_in_': self.n_features_in_,
            'n_samples_seen_': self.n_samples_seen_,
            'is_fitted': self.is_fitted,
            '_XTX_inv': self._XTX_inv,
            'device': str(self.device),
            'model_class': self.__class__.__name__
        }
        
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.coef_ = state['coef_']
        self.intercept_ = state['intercept_']
        self.fit_intercept = state['fit_intercept']
        self.n_features_in_ = state['n_features_in_']
        self.n_samples_seen_ = state['n_samples_seen_']
        self.is_fitted = state['is_fitted']
        self._XTX_inv = state.get('_XTX_inv')
    
    def to(self, device: Union[str, torch.device]):
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
        
        Returns:
            self: Model on new device
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        return self
    
# ============================================================================
# LOGISTIC REGRESSION
# ============================================================================


class LogisticRegressionNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.0, **kwargs):
        super().__init__()
        # For binary classification, output single logit (not 2)
        output_size = 1 if n_classes == 2 else n_classes
        
        layers = [nn.Linear(n_features, output_size)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.linear = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.linear(x)
        # For binary: apply sigmoid to get probability of positive class
        # For multiclass: return raw logits (CrossEntropyLoss expects logits)
        return logits


class LogisticRegression(BaseNeuralModel):
    """
    Logistic Regression using PyTorch with GPU support.
    
    Supports binary and multi-class classification.
    
    Args:
        epochs: Number of training epochs (default: 100)
        lr: Learning rate (default: 0.01)
        batch_size: Batch size for training (default: 32)
        optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
        device: Computation device
    """
    
    def __init__(self, 
                 epochs: int = 100,
                 lr: float = 0.01,
                 batch_size: int = 32,
                 optimizer: str = 'adam',
                 device: Optional[torch.device] = None):
        super().__init__(epochs, lr, batch_size, optimizer, device)
        self.n_classes_ = None
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'LogisticRegression':
        """
        Fit logistic regression model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            verbose: Print training progress
            
        Returns:
            self: Fitted model
        """
        # Convert to tensors
        X = self._to_tensor(X, dtype=torch.float32)
        y_np = y if isinstance(y, np.ndarray) else self._to_numpy(y)
        
        # Get unique classes
        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        
        # Convert labels to tensor
        y = self._to_tensor(y_np, dtype=torch.long)
        
        # Validate input
        self._validate_input(X, y)
        
        # Initialize model
        n_features = X.shape[1]
        self.model = LogisticRegressionNet(n_features, self.n_classes_).to(self.device)
        
        # Setup optimizer and loss
        self._create_optimizer()
        
        if self.n_classes_ == 2:
            # Binary classification: BCEWithLogitsLoss (more numerically stable)
            self.criterion = nn.BCEWithLogitsLoss()
            # Convert labels to float for binary classification
            y = y.float().unsqueeze(1)
        else:
            # Multi-class: CrossEntropyLoss expects class indices
            self.criterion = nn.CrossEntropyLoss()
        
        # Create data loader
        data_loader = self._create_data_loader(X, y, shuffle=True)
        
        # Training loop
        self._training_history = {'loss': [], 'epoch': []}
        
        for epoch in range(self.epochs):
            avg_loss = self._train_epoch(data_loader)
            
            self._training_history['loss'].append(avg_loss)
            self._training_history['epoch'].append(epoch)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Extract coefficients
        with torch.no_grad():
            if self.n_classes_ == 2:
                # Binary: model has 1 output
                self.coef_ = self.model.linear[0].weight.cpu().numpy()
                self.intercept_ = self.model.linear[0].bias.cpu().numpy()
            else:
                # Multi-class: model has n_classes outputs
                self.coef_ = self.model.linear[0].weight.cpu().numpy()
                self.intercept_ = self.model.linear[0].bias.cpu().numpy()
        
        # Update metadata
        self._update_fit_info(X)
        
        return self
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        with torch.no_grad():
            X = self._to_tensor(X, dtype=torch.float32)
            logits = self.model(X)
            
            if self.n_classes_ == 2:
                # Binary: apply sigmoid to get probability of positive class
                proba_pos = torch.sigmoid(logits)
                proba_neg = 1 - proba_pos
                probas = torch.cat([proba_neg, proba_pos], dim=1)
            else:
                # Multi-class: apply softmax
                probas = torch.softmax(logits, dim=1)
            
            return self._to_numpy(probas)
    
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
        
        # Map back to original class labels
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
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_,
            'n_samples_seen_': self.n_samples_seen_,
            'is_fitted': self.is_fitted,
            'epochs': self.epochs,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'optimizer_name': self.optimizer_name,
            'training_history': self._training_history,
            'device': str(self.device),
            'model_class': self.__class__.__name__
        }
        
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore hyperparameters
        self.epochs = state['epochs']
        self.lr = state['lr']
        self.batch_size = state['batch_size']
        self.optimizer_name = state['optimizer_name']
        self.classes_ = state['classes_']
        self.n_classes_ = state['n_classes_']
        self.n_features_in_ = state['n_features_in_']
        self.n_samples_seen_ = state['n_samples_seen_']
        self.is_fitted = state['is_fitted']
        self.coef_ = state['coef_']
        self.intercept_ = state['intercept_']
        self._training_history = state['training_history']
        
        # Recreate model
        self.model = LogisticRegressionNet(self.n_features_in_, self.n_classes_).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
    
    def to(self, device: Union[str, torch.device]):
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
        
        Returns:
            self: Model on new device
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self

# ============================================================================
# RIDGE REGRESSION (L2 Regularization)
# ============================================================================

class RidgeRegression(BaseNeuralModel):
    """
    Ridge Regression (Linear Regression with L2 regularization).
    
    Args:
        alpha: Regularization strength (default: 1.0)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        optimizer: Optimizer type
        device: Computation device
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 epochs: int = 100,
                 lr: float = 0.01,
                 batch_size: int = 32,
                 optimizer: str = 'adam',
                 device: Optional[torch.device] = None):
        super().__init__(epochs, lr, batch_size, optimizer, device)
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'RidgeRegression':
        """Fit ridge regression model."""
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y, dtype=torch.float32)
        
        if y.dim() > 1:
            y = y.squeeze()
        
        self._validate_input(X, y)
        
        # Initialize model
        n_features = X.shape[1]
        self.model = LinearRegressionNet(n_features).to(self.device)
        
        # Setup optimizer and loss
        self._create_optimizer()
        self.criterion = nn.MSELoss()
        
        # Reshape y for training
        y = y.unsqueeze(1)
        
        # Create data loader
        data_loader = self._create_data_loader(X, y, shuffle=True)
        
        # Training loop
        self._training_history = {'loss': [], 'epoch': []}
        
        for epoch in range(self.epochs):
            avg_loss = self._train_epoch(data_loader)
            
            self._training_history['loss'].append(avg_loss)
            self._training_history['epoch'].append(epoch)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Extract coefficients
        with torch.no_grad():
            self.coef_ = self.model.linear.weight.cpu().numpy().flatten()
            self.intercept_ = self.model.linear.bias.cpu().numpy()[0]
        
        self._update_fit_info(X)
        
        return self
    
    def _train_epoch(self, data_loader) -> float:
        """Train for one epoch with L2 regularization."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in data_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Add L2 regularization
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.model.parameters():
                l2_reg += torch.norm(param, 2)
            loss = loss + self.alpha * l2_reg
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict target values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        with torch.no_grad():
            X = self._to_tensor(X, dtype=torch.float32)
            predictions = self.model(X)
            return self._to_numpy(predictions).flatten()
    
    def score(self, X: Union[np.ndarray, torch.Tensor],
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        y = y.flatten()
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1.0 - (ss_res / ss_tot)
    
    def get_params(self, deep: bool = True):
        params = super().get_params(deep=deep)
        params.update({
            'alpha': self.alpha,
            'coef_': self.coef_,
            'intercept_': self.intercept_
        })
        return params
    
    def save(self, filepath: str):
        """Save model to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'alpha': self.alpha,
            'n_features_in_': self.n_features_in_,
            'n_samples_seen_': self.n_samples_seen_,
            'is_fitted': self.is_fitted,
            'epochs': self.epochs,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'optimizer_name': self.optimizer_name,
            'training_history': self._training_history,
            'device': str(self.device),
            'model_class': self.__class__.__name__
        }
        
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.alpha = state['alpha']
        self.epochs = state['epochs']
        self.lr = state['lr']
        self.batch_size = state['batch_size']
        self.optimizer_name = state['optimizer_name']
        self.n_features_in_ = state['n_features_in_']
        self.n_samples_seen_ = state['n_samples_seen_']
        self.is_fitted = state['is_fitted']
        self.coef_ = state['coef_']
        self.intercept_ = state['intercept_']
        self._training_history = state['training_history']
        
        # Recreate model
        self.model = LinearRegressionNet(self.n_features_in_).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
    
    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self


# ============================================================================
# LASSO REGRESSION (L1 Regularization)
# ============================================================================

class LassoRegression(BaseNeuralModel):
    """
    Lasso Regression (Linear Regression with L1 regularization).
    
    Args:
        alpha: Regularization strength (default: 1.0)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        optimizer: Optimizer type
        device: Computation device
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 epochs: int = 100,
                 lr: float = 0.01,
                 batch_size: int = 32,
                 optimizer: str = 'adam',
                 device: Optional[torch.device] = None):
        super().__init__(epochs, lr, batch_size, optimizer, device)
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'LassoRegression':
        """Fit lasso regression model."""
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y, dtype=torch.float32)
        
        if y.dim() > 1:
            y = y.squeeze()
        
        self._validate_input(X, y)
        
        # Initialize model
        n_features = X.shape[1]
        self.model = LinearRegressionNet(n_features).to(self.device)
        
        # Setup optimizer and loss
        self._create_optimizer()
        self.criterion = nn.MSELoss()
        
        # Reshape y for training
        y = y.unsqueeze(1)
        
        # Create data loader
        data_loader = self._create_data_loader(X, y, shuffle=True)
        
        # Training loop
        self._training_history = {'loss': [], 'epoch': []}
        
        for epoch in range(self.epochs):
            avg_loss = self._train_epoch(data_loader)
            
            self._training_history['loss'].append(avg_loss)
            self._training_history['epoch'].append(epoch)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Extract coefficients
        with torch.no_grad():
            self.coef_ = self.model.linear.weight.cpu().numpy().flatten()
            self.intercept_ = self.model.linear.bias.cpu().numpy()[0]
        
        self._update_fit_info(X)
        
        return self
    
    def _train_epoch(self, data_loader) -> float:
        """Train for one epoch with L1 regularization."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in data_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Add L1 regularization
            l1_reg = torch.tensor(0., device=self.device)
            for param in self.model.parameters():
                l1_reg += torch.norm(param, 1)
            loss = loss + self.alpha * l1_reg
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict target values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        with torch.no_grad():
            X = self._to_tensor(X, dtype=torch.float32)
            predictions = self.model(X)
            return self._to_numpy(predictions).flatten()
    
    def score(self, X: Union[np.ndarray, torch.Tensor],
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        y = y.flatten()
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1.0 - (ss_res / ss_tot)
    
    def get_params(self, deep: bool = True):
        """Return model parameters."""
        params = super().get_params(deep=deep)
        params.update({
            'alpha': self.alpha,
            'coef_': self.coef_,
            'intercept_': self.intercept_
        })
        return params

    def save(self, filepath: str):
        """Save model to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'alpha': self.alpha,
            'n_features_in_': self.n_features_in_,
            'n_samples_seen_': self.n_samples_seen_,
            'is_fitted': self.is_fitted,
            'epochs': self.epochs,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'optimizer_name': self.optimizer_name,
            'training_history': self._training_history,
            'device': str(self.device),
            'model_class': self.__class__.__name__
        }
        
        torch.save(state, filepath)

    def load(self, filepath: str):
        """Load model from file."""
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.alpha = state['alpha']
        self.epochs = state['epochs']
        self.lr = state['lr']
        self.batch_size = state['batch_size']
        self.optimizer_name = state['optimizer_name']
        self.n_features_in_ = state['n_features_in_']
        self.n_samples_seen_ = state['n_samples_seen_']
        self.is_fitted = state['is_fitted']
        self.coef_ = state['coef_']
        self.intercept_ = state['intercept_']
        self._training_history = state['training_history']
        
        # Recreate model
        self.model = LinearRegressionNet(self.n_features_in_).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self
    





# def test_linear_regression():
#     print("\n=== Testing Linear Regression ===")
#     X = np.random.randn(100, 3)
#     y = X @ np.array([1.5, -2.0, 3.0]) + 4.0  # y = 1.5x1 - 2x2 + 3x3 + 4
#     model = LinearRegression(fit_intercept=True)
    
#     model.fit(X, y, verbose=True)
#     preds = model.predict(X)
#     score = model.score(X, y)
#     print(f"R² Score: {score:.4f}")
#     print(f"Intercept: {model.intercept_:.4f}")
#     print(f"Coefficients: {model.coef_}")
    
#     # Test save/load
#     path = "linear_model.pt"
#     model.save(path)
#     new_model = LinearRegression()
#     new_model.load(path)
#     os.remove(path)
#     print("✅ Linear Regression test passed.")


# def test_ridge_regression():
#     print("\n=== Testing Ridge Regression ===")
#     X = np.random.randn(120, 5)
#     y = X @ np.array([2.0, -1.0, 0.5, 3.0, -2.0]) + 1.0
#     model = RidgeRegression(alpha=0.1, epochs=50, lr=0.01)
    
#     model.fit(X, y, verbose=True)
#     preds = model.predict(X)
#     score = model.score(X, y)
#     print(f"R² Score: {score:.4f}")
#     print(f"Intercept: {model.intercept_:.4f}")
#     print(f"Coefficients: {model.coef_}")
    
#     # Test save/load
#     path = "ridge_model.pt"
#     model.save(path)
#     new_model = RidgeRegression()
#     new_model.load(path)
#     os.remove(path)
#     print("✅ Ridge Regression test passed.")


# def test_logistic_regression_binary():
#     print("\n=== Testing Logistic Regression (Binary) ===")
#     X = np.random.randn(200, 4)
#     y = (X[:, 0] + X[:, 1] * 2.0 + np.random.randn(200)) > 0
#     y = y.astype(int)
    
#     model = LogisticRegression(epochs=100, lr=0.05, batch_size=32)
#     model.fit(X, y, verbose=True)
#     preds = model.predict(X)
#     acc = model.score(X, y)
#     print(f"Accuracy: {acc:.4f}")
#     print(f"Classes: {model.classes_}")
    
#     # Test save/load
#     path = "logistic_model.pt"
#     model.save(path)
#     new_model = LogisticRegression()
#     new_model.load(path)
#     os.remove(path)
#     print("✅ Logistic Regression (Binary) test passed.")


# def test_logistic_regression_multiclass():
#     print("\n=== Testing Logistic Regression (Multi-Class) ===")
#     X = np.random.randn(300, 4)
#     y = np.random.choice([0, 1, 2], size=300)
    
#     model = LogisticRegression(epochs=80, lr=0.05, batch_size=32)
#     model.fit(X, y, verbose=True)
#     preds = model.predict(X)
#     acc = model.score(X, y)
#     print(f"Accuracy: {acc:.4f}")
#     print(f"Classes: {model.classes_}")
    
#     # Test save/load
#     path = "logistic_multiclass.pt"
#     model.save(path)
#     new_model = LogisticRegression()
#     new_model.load(path)
#     os.remove(path)
#     print("✅ Logistic Regression (Multi-Class) test passed.")


# if __name__ == "__main__":
#     torch.manual_seed(42)
#     np.random.seed(42)

#     test_linear_regression()
#     test_ridge_regression()
#     test_logistic_regression_binary()
#     test_logistic_regression_multiclass()
#     print("\n🎯 All regression model tests completed successfully.")


# PS D:\Auto_ML\new_auto_ml> & C:/Users/Smile/anaconda3/envs/tensorflow/python.exe d:/Auto_ML/new_auto_ml/lightning_ml/regression.py

# === Testing Linear Regression ===
# LinearRegression fitted with 3 features on 100 samples
# Intercept: 4.0000
# Coefficients shape: (3,)
# R² Score: 1.0000
# Intercept: 4.0000
# Coefficients: [ 1.5 -2.   3. ]
# ✅ Linear Regression test passed.

# === Testing Ridge Regression ===
# Epoch [10/50], Loss: 9.7488
# Epoch [20/50], Loss: 6.3802
# Epoch [30/50], Loss: 4.1420
# Epoch [40/50], Loss: 2.6785
# Epoch [50/50], Loss: 1.8854
# R² Score: 0.9135
# Intercept: 1.0073
# Coefficients: [ 1.6411462  -0.78659725  0.2415431   1.989736   -1.4593322 ]
# ✅ Ridge Regression test passed.

# === Testing Logistic Regression (Binary) ===
# Epoch [10/100], Loss: 0.3248
# Epoch [20/100], Loss: 0.3044
# Epoch [30/100], Loss: 0.3058
# Epoch [40/100], Loss: 0.2946
# Epoch [50/100], Loss: 0.2620
# Epoch [60/100], Loss: 0.3219
# Epoch [70/100], Loss: 0.2587
# Epoch [80/100], Loss: 0.3204
# Epoch [90/100], Loss: 0.3146
# Epoch [100/100], Loss: 0.2759
# Accuracy: 0.8650
# Classes: [0 1]
# ✅ Logistic Regression (Binary) test passed.

# === Testing Logistic Regression (Multi-Class) ===
# Epoch [10/80], Loss: 1.0943
# Epoch [20/80], Loss: 1.0834
# Epoch [30/80], Loss: 1.0955
# Epoch [40/80], Loss: 1.0981
# Epoch [50/80], Loss: 1.0828
# Epoch [60/80], Loss: 1.0899
# Epoch [70/80], Loss: 1.0821
# Epoch [80/80], Loss: 1.0836
# Accuracy: 0.4100
# Classes: [0 1 2]
# ✅ Logistic Regression (Multi-Class) test passed.

# 🎯 All regression model tests completed successfully.
# PS D:\Auto_ML\new_auto_ml>


